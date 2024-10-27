
import torch
from contextlib import nullcontext
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import time
import wandb
import torch.cuda.profiler as profiler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, LambdaLR

import torch._dynamo

torch._dynamo.config.suppress_errors = True  # TODO check if this can be removed

# import modules
import os

from modulus.models.graphcast.graph_cast_net import GraphCastNet
from modulus.utils.graphcast.loss import (
    CellAreaWeightedLossFunction,
    GraphCastLossFunction,
)
from modulus.launch.logging import (
    PythonLogger,
    initialize_wandb,
    RankZeroLoggingWrapper,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint

from train_utils import count_trainable_params, prepare_input
from loss.utils import normalized_grid_cell_area
from train_base import BaseTrainer
from validation_base import Validation
from modulus.datapipes.climate import ERA5HDF5Datapipe, SyntheticWeatherDataLoader
from modulus.distributed import DistributedManager
from modulus.utils.graphcast.data_utils import StaticData

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig




@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    model = GraphCastNet(
                # mesh_level=cfg.mesh_level,
                multimesh=cfg.multimesh,
                input_res=tuple(cfg.latlon_res),
                input_dim_grid_nodes=(
                    cfg.num_channels_climate
                    + cfg.use_cos_zenith
                    + 4 * cfg.use_time_of_year_index
                )
                * (cfg.num_history + 1)
                + cfg.num_channels_static,
                input_dim_mesh_nodes=3,
                input_dim_edges=4,
                output_dim_grid_nodes=cfg.num_channels_climate,
                processor_type=cfg.processor_type,
                khop_neighbors=cfg.khop_neighbors,
                num_attention_heads=cfg.num_attention_heads,
                processor_layers=cfg.processor_layers,
                hidden_dim=cfg.hidden_dim,
                norm_type=cfg.norm_type,
                do_concat_trick=cfg.concat_trick,
                use_cugraphops_encoder=cfg.cugraphops_encoder,
                use_cugraphops_processor=cfg.cugraphops_processor,
                use_cugraphops_decoder=cfg.cugraphops_decoder,
                recompute_activation=cfg.recompute_activation,
            )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    scheduler1 = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=cfg.num_iters_step1,
    )
    scheduler2 = CosineAnnealingLR(
        optimizer, T_max=cfg.num_iters_step2, eta_min=0.0
    )
    scheduler3 = LambdaLR(
        optimizer, lr_lambda=lambda epoch: (cfg.lr_step3 / cfg.lr)
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2, scheduler3],
        milestones=[cfg.num_iters_step1, cfg.num_iters_step1 + cfg.num_iters_step2],
    )
    scaler = GradScaler(enabled=False)
    if torch.cuda.is_available():
        device = torch.device("cuda",0)
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")    
    dtype = torch.float32

    model=model.to(device).to(dtype)

    load_checkpoint(
        to_absolute_path(cfg.ckpt_path),
        models=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
    )
    
        
    import onnx, onnxruntime
    onnx_name = "model.onnx"
    # model = model.eval().cpu()
    with torch.no_grad():
        torch_input = torch.randn([1, 23, 721, 1440]).to(device).to(dtype)

        print()
        torch.onnx.export(
            model.cpu(),
            torch_input,
            onnx_name,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            opset_version=18,
            verbose=True,
        )
        # ss=onnx.load(onnx_name)
        # print(ss)
        torch_outputs = model(torch_input)
        if isinstance(torch_input, torch.Tensor):
            torch_input = (torch_input,)

        ort_session = onnxruntime.InferenceSession("./model.onnx", providers=['CPUExecutionProvider'])
        ort_inputs = {inp.name: v.detach().cpu().numpy()
            for inp, v in zip(ort_session.get_inputs(), torch_input)}
        onnxruntime_outputs = ort_session.run(None, ort_inputs)   
        np.testing.assert_allclose(onnxruntime_outputs[0], torch_outputs.detach().cpu().numpy(), rtol=1e-03, atol=1e-05)
    # # 导出模型前，必须调用model.eval()或者model.train(False)
    # with torch.no_grad():
    #     torch_input = torch.randn([1, 23, 721, 1440]).to(device).to(dtype)
    #     onnx_program = torch.onnx.dynamo_export(model, torch_input)
    #     onnx_program.save("my_image_classifier.onnx")
    #     import onnx, onnxruntime
    #     onnx_model = onnx.load("my_image_classifier.onnx")
    #     onnx.checker.check_model(onnx_model)
    #     onnx_input = onnx_program.adapt_torch_inputs_to_onnx(torch_input)
    #     print(f"Input length: {len(onnx_input)}")
    #     print(f"Sample input: {onnx_input}")

    #     ort_session = onnxruntime.InferenceSession("./my_image_classifier.onnx", providers=['CUDAExecutionProvider'])

    #     onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}

    #     onnxruntime_outputs = ort_session.run(None, onnxruntime_input)    
    #     torch_outputs = model(torch_input)
    #     torch_outputs = onnx_program.adapt_torch_outputs_to_onnx(torch_outputs)

    #     assert len(torch_outputs) == len(onnxruntime_outputs)
    #     for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    #         torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

    #     print("PyTorch and ONNX Runtime output matched!")
    #     print(f"Output length: {len(onnxruntime_outputs)}")
    #     print(f"Sample output: {onnxruntime_outputs}")
        
        
        
def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        
if __name__ == "__main__":
    main()
