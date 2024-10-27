
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
    model_f32 = GraphCastNet(
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
                norm_type='LayerNorm',
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
        
    dtype = torch.float32
    model=model.to(device).to(dtype=torch.bfloat16)
    model_f32=model_f32.to(device).to(dtype)

    load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
    # model=model.to(device).to(dtype=torch.bfloat16)

    load_checkpoint(
        to_absolute_path(cfg.ckpt_path),
        models=model_f32,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
    )
    model.eval()
    model_f32.eval()
    dummy_input=torch.empty([1, 23, 721, 1440]).to(dtype=dtype).to(device=device)
    torch.save(model_f32,'model_f32.pth')
    model_save = torch.load('/workspaces/AIWP_modulus/outputs/graphcast/model_f32.pth')
    # 导出模型前，必须调用model.eval()或者model.train(False)
    with torch.no_grad():

        # # 导出模型
        onnx_file_name = "accumodel_test_f32.onnx"
        torch.onnx.export(model_f32,        # 模型的名称
                        dummy_input,   # 一组实例化输入
                        onnx_file_name,   # 文件保存路径/名称
                        export_params=True,        #  如果指定为True或默认, 参数也会被导出. 如果你要导出一个没训练过的就设为 False.         # ONNX 算子集的版本，当前已更新到15
                        #do_constant_folding=True,  # 是否执行常量折叠优化
                        input_names = ['input'],   # 输入模型的张量的名称
                        output_names = ['output'], # 输出模型的张量的名称
                        opset_version=20
                        # dynamic_axes将batch_size的维度指定为动态，
                        # 后续进行推理的数据可以与导出的dummy_input的batch_size不同
                        # dynamic_axes={'input' : {0 : 'batch_size'},    
                        #                 'output' : {0 : 'batch_size'}}
                        )
        x = torch.randn(1, 23, 721, 1440, requires_grad=False).to(dtype=dtype).to(device)
        output_f32 = model_f32(x)
        output_f32_sss = model_save(x)
        x2 = x.to(dtype=torch.bfloat16).to(device)

        output = model(x2)
        output=output.to(dtype=dtype)
        # np.testing.assert_allclose(output.detach().cpu().numpy(), output_f32.detach().cpu().numpy(), rtol=1e-03, atol=1e-05)


        import onnxruntime as ort
        # new added: onnx model comparison
        onnx_filename='/workspaces/AIWP_modulus/outputs/graphcast/accumodel_test_f32.onnx'
        # providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),
        #                                     "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]
        sess_options = ort.SessionOptions()
        accu_model=ort.InferenceSession(onnx_filename)
            # load normalisation values
        ort_inputs = {accu_model.get_inputs()[0].name:x.detach().cpu().numpy()}
        outpred_onnx = accu_model.run(None, ort_inputs) 
        sss=output.detach().cpu().numpy()
        np.testing.assert_allclose(output.detach().cpu().numpy(), outpred_onnx[0], rtol=1e-03, atol=1e-05)
        print('end')
        
def test():
    import onnx
        # 加载模型
    try:
    # 当我们的模型不可用时，将会报出异常
        onnx.checker.check_model('outputs/graphcast/accumodel_test_f32.onnx')
    except onnx.checker.ValidationError as e:
        print("The model is invalid: %s"%e)
    else:
        # 模型可用时，将不会报出异常，并会输出“The model is valid!”
        print("The model is valid!")
    #print(model)

    

if __name__ == "__main__":
    main()
    test()