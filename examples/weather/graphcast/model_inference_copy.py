
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
    if torch.cuda.is_available():
        device = torch.device("cuda",0)
    else:
        device = torch.device("cpu")
    
    model=torch.load('/workspaces/AIWP_modulus/outputs/graphcast/Accumodel_small_bf16.pth')
    model=model.to(device).to(dtype=torch.bfloat16)
    model.eval()
    x = torch.randn(1, 95, 721, 1440, requires_grad=False).to(dtype=torch.bfloat16).to(device)
    with torch.no_grad():
        start_time = time.time()
        y=model(x)
        end_time = time.time()
        execution_time = end_time - start_time
        start_time=end_time
        print(f"程序执行时间: {execution_time} 秒")  
    print(1)
    
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    
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
        device = torch.device("cuda",1)
    else:
        device = torch.device("cpu")
        
    dtype = torch.float32
    model=model.to(device).to(dtype=torch.bfloat16)

    load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
    # model=model.to(device).to(dtype=torch.bfloat16)

    model.eval()
    torch.save(model,'Accumodel_small_bf16.pth')
    
    

if __name__ == "__main__":
    
    main()
