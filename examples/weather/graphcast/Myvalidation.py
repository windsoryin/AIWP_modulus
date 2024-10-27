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
from modulus.datapipes.climate import ERA5HDF5Datapipe, SyntheticWeatherDataLoader
from modulus.distributed import DistributedManager
from modulus.utils.graphcast.data_utils import StaticData

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig


import os
import torch
import matplotlib.pyplot as plt

from modulus.datapipes.climate import ERA5HDF5Datapipe

from train_utils import prepare_input


import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


class Validation:
    """Run validation on GraphCast model"""

    def __init__(self, cfg: DictConfig, model, dtype, dist, static_data):
        self.val_dir = to_absolute_path(cfg.val_dir)
        self.model = model
        self.dtype = dtype
        self.dist = dist
        self.static_data = static_data
        self.interpolation_type = (
            "INTERP_LINEAR" if cfg.latlon_res != (721, 1440) else None
        )  # interpolate if not in native resolution
        self.cos_zenith_args = {
            "dt": 6.0,
            "start_year": 2017,
        }
        self.val_datapipe = ERA5HDF5Datapipe(
            data_dir=os.path.join(cfg.dataset_path, "inference_data"),
            stats_dir=os.path.join(cfg.dataset_path, "stats/model_large_135"),
            channels=[i for i in range(cfg.num_channels_climate)],
            latlon_resolution=cfg.latlon_res,
            interpolation_type=self.interpolation_type,
            num_steps=cfg.num_val_steps,
            num_history=cfg.num_history,
            use_cos_zenith=cfg.use_cos_zenith,
            use_time_of_year_index=cfg.use_time_of_year_index,
            cos_zenith_args=self.cos_zenith_args,
            batch_size=1,
            num_samples_per_year=cfg.num_val_spy,
            shuffle=False,
            device=self.dist.device,
            process_rank=self.dist.rank,
            world_size=self.dist.world_size,
            num_workers=cfg.num_workers,
        )
        print(f"Loaded validation datapipe of size {len(self.val_datapipe)}")
        self.num_history = cfg.num_history
        self.stride = cfg.stride
        self.dt = cfg.dt
        self.num_samples_per_year_train = cfg.num_samples_per_year_train

    @torch.no_grad()
    def step(self, channels=[0, 1, 2], iter=0, time_idx=None):
        torch.cuda.nvtx.range_push("Validation")
        os.makedirs(self.val_dir, exist_ok=True)
        loss_epoch = 0
        prepare_input_vars = {
            "num_history": self.num_history,
            "static_data": self.static_data,
            "stride": self.stride,
            "dt": self.dt,
            "num_samples_per_year": self.num_samples_per_year_train,
            "device": self.dist.device,
        }
        for i, data in enumerate(self.val_datapipe):
            invar = data[0]["invar"]
            outvar = data[0]["outvar"][0]
            try:
                cos_zenith = data[0]["cos_zenith"]
            except KeyError:
                cos_zenith = None
            try:
                time_idx = data[0]["time_of_year_idx"].item()
            except KeyError:
                time_idx = None
            invar_cat = prepare_input(
                invar=invar,
                cos_zenith=cos_zenith,
                time_idx=time_idx,
                **prepare_input_vars,
                step=1,
            )
            invar_cat = invar_cat.to(dtype=self.dtype)

            pred = (
                torch.empty(outvar.shape)
                .to(dtype=self.dtype)
                .to(device=self.dist.device)
            )
            for t in range(outvar.shape[0]):
                outpred = self.model(invar_cat)
                pred[t] = outpred
                if self.num_history > 0:
                    # drop the first time step, and append the prediction as the last time step in invar
                    invar = torch.cat((invar[:, 1:, :, :], outpred.unsqueeze(1)), dim=1)
                else:
                    invar = outpred
                
                time_idx=time_idx+1 # 0928 add: next time step
                invar_cat = prepare_input(
                    invar=invar,
                    cos_zenith=cos_zenith,
                    time_idx=time_idx,
                    **prepare_input_vars,
                    step=t + 2,
                )
                invar_cat = invar_cat.to(dtype=self.dtype)

            loss_epoch += torch.mean(torch.pow(pred - outvar, 2),dim=(2,3))
            torch.cuda.nvtx.range_pop()

            pred = pred.to(torch.float32).cpu().numpy()
            outvar = outvar.to(torch.float32).cpu().numpy()

            del invar, outpred
            torch.cuda.empty_cache()

            if i == 0:
                for c_i,chan in enumerate(channels):
                    plt.close("all")
                    fig, ax = plt.subplots(3, pred.shape[0], figsize=(15, 5))
                    fig.subplots_adjust(hspace=0.5, wspace=0.3)

                    for t in range(outvar.shape[0]):
                        im_pred = ax[0, t].imshow(pred[t, chan], vmin=-1.5, vmax=1.5)
                        ax[0, t].set_title(f"Prediction (t={t+1})", fontsize=10)
                        fig.colorbar(
                            im_pred, ax=ax[0, t], orientation="horizontal", pad=0.4
                        )

                        im_outvar = ax[1, t].imshow(
                            outvar[t, chan], vmin=-1.5, vmax=1.5
                        )
                        ax[1, t].set_title(f"Ground Truth (t={t+1})", fontsize=10)
                        fig.colorbar(
                            im_outvar, ax=ax[1, t], orientation="horizontal", pad=0.4
                        )

                        im_diff = ax[2, t].imshow(
                            abs(pred[t, chan] - outvar[t, chan]), vmin=0.0, vmax=0.5
                        )
                        ax[2, t].set_title(f"Abs. Diff. (t={t+1})", fontsize=10)
                        fig.colorbar(
                            im_diff, ax=ax[2, t], orientation="horizontal", pad=0.4
                        )

                    fig.savefig(
                        os.path.join(
                            self.val_dir,
                            f"era5_validation_channel{chan}_iter{iter}.png",
                        )
                    )
                    wandb.log({f"val_chan{chan}_iter{iter}": fig}, step=iter)

        return loss_epoch.to(torch.float32).cpu().numpy() / len(self.val_datapipe)






@hydra.main(version_base="1.3", config_path="conf", config_name="config_val")
def main(cfg: DictConfig) -> None:
        # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # initialize loggers at rank 0
    if dist.rank == 0: 
        initialize_wandb(
            project="GraphCast",
            entity="Modulus",
            name=f"GraphCast-{HydraConfig.get().job.name}",
            group="group",
            mode=cfg.wb_mode,
        )  # Wandb logger
    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    rank_zero_logger.file_logging()
    
    
    
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
    # if torch.cuda.is_available():
    #     device = torch.device("cuda",0)
    # else:
    #     device = torch.device("cpu")
        
    dtype=torch.bfloat16
    
    model=model.to(dist.device).to(dtype=torch.bfloat16)
    load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=dist.device,
        )
    

    static_dataset_path = (
            to_absolute_path(cfg.static_dataset_path)
            if cfg.static_dataset_path
            else None
        )
    # Get the static data
    
    latitudes = torch.linspace(-90, 90, steps=721)
    longitudes = torch.linspace(-180, 180, steps=1441)[1:]
    static_data = StaticData(
                static_dataset_path, latitudes, longitudes
            ).get()
    static_data = static_data.to(dist.device)
    
    # initialize validation class
    validation = Validation(
                cfg, model, dtype, dist, static_data
            )
    
    iter=1
    loss_epoch=validation.step(channels=list(np.arange(cfg.num_channels_val)), iter=iter)
    # Ensure loss_epoch is moved to CPU and converted to numpy for plotting
    # loss_epoch = loss_epoch.cpu()

    for chan in range(cfg.num_channels_val):
        fig = plt.figure()
        plt.plot(loss_epoch[:, chan])
        plt.title(f"Loss for Channel {chan}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(validation.val_dir, f"loss_epoch_{chan}.png"))
        plt.close(fig)
                    





if __name__ == "__main__":
    main()