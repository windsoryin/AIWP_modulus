# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda.amp import autocast

import sys

from train_utils import prepare_input # add to rollout prediction


class BaseTrainer:
    """Trainer class"""

    def __init__(self):
        pass

    def rollout(self, grid_nfeat, y, cos_zenith=None, time_idx=None):
        with autocast(enabled=self.amp, dtype=self.amp_dtype):
            total_loss = 0
            pred_prev = grid_nfeat
            for i in range(y.size(dim=1)):
                # Shape of y is [N, M, C, H, W]. M is the number of steps
                pred = self.model(pred_prev)
                loss = self.criterion(pred, y[:, i])
                total_loss += loss
                pred_prev = pred
            return total_loss

    def forward(self, grid_nfeat, y):
        # forward pass
        torch.cuda.nvtx.range_push("Loss computation")
        if self.pyt_profiler:
            with profile(
                activities=[ProfilerActivity.CUDA], record_shapes=True
            ) as prof:
                with record_function("training_step"):
                    loss = self.rollout(grid_nfeat, y)

            print(
                prof.key_averages(group_by_input_shape=True).table(
                    sort_by="cuda_time_total", row_limit=10
                )
            )
            exit(0)
        else:
            loss = self.rollout(grid_nfeat, y)
        torch.cuda.nvtx.range_pop()
        return loss
    
    def backward(self, loss):
        # backward pass
        torch.cuda.nvtx.range_push("Weight gradients")
        if self.amp:
            self.scaler.scale(loss).backward()
            torch.cuda.nvtx.range_pop()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            torch.cuda.nvtx.range_pop()
            self.optimizer.step()

    def train(self, grid_nfeat, y):
        self.optimizer.zero_grad()
        loss = self.forward(grid_nfeat, y)
        self.backward(loss)
        self.scheduler.step()
        return loss
    
    def autogress_train(self, grid_nfeat, y, cos_zenith, time_idx, prepare_input_vars,clear_flag=False):
        self.optimizer.zero_grad()
        loss = self.autogress_forward(grid_nfeat, y, cos_zenith, time_idx, prepare_input_vars,clear_flag)
        self.backward(loss)
        self.scheduler.step()
        return loss
    
    def autogress_forward(self, grid_nfeat, y, cos_zenith, time_idx, prepare_input_vars,clear_flag=False):
        # forward pass
        torch.cuda.nvtx.range_push("Loss computation")
        if self.pyt_profiler:
            with profile(
                activities=[ProfilerActivity.CUDA], record_shapes=True
            ) as prof:
                with record_function("training_step"):
                    loss = self.autogress_rollout(grid_nfeat, y, cos_zenith, time_idx, prepare_input_vars,clear_flag)

            print(
                prof.key_averages(group_by_input_shape=True).table(
                    sort_by="cuda_time_total", row_limit=10
                )
            )
            exit(0)
        else:
            loss = self.autogress_rollout(grid_nfeat, y, cos_zenith, time_idx, prepare_input_vars,clear_flag)
        torch.cuda.nvtx.range_pop()
        return loss
    
    def autogress_rollout(self, grid_nfeat, y, cos_zenith, time_idx, prepare_input_vars,clear_flag=False):
        
        with autocast(enabled=self.amp, dtype=self.amp_dtype):
            total_loss = 0
            invar = grid_nfeat
            outvar=y
            invar_cat = prepare_input(
                        invar=invar,
                        cos_zenith=cos_zenith,
                        time_idx=time_idx,
                        **prepare_input_vars,
                        step=1,
                    )
            invar_cat, outvar = invar_cat.to(dtype=self.dtype), outvar.to(
                dtype=self.dtype
            )
            for i in range(outvar.size(dim=1)):
                # Shape of y is [N, M, C, H, W]. M is the number of steps
                pred = self.model(invar_cat)
                loss = self.criterion(pred, outvar[:, i])
                total_loss += loss
                step=i + 2
                if y.size(dim=1)==1:
                    # 0928 add: if only one time step, return
                    if clear_flag:
                        print('clear_flag')
                        del invar, invar_cat, outvar
                        torch.cuda.empty_cache()
                    return total_loss
                else:
                    if prepare_input_vars['num_history'] > 0:
                        # drop the first time step, and append the prediction as the last time step in invar
                        invar = torch.cat((invar[:, 1:, :, :], pred.unsqueeze(1)), dim=1)
                    else:
                        invar = pred
    
                    time_idx=time_idx+1 # 0928 add: next time step
                    invar_cat = prepare_input(
                        invar=invar,
                        cos_zenith=cos_zenith,
                        time_idx=time_idx,
                        **prepare_input_vars,
                        step=step,
                    )
                    invar_cat = invar_cat.to(dtype=self.dtype)
            if clear_flag:
                print('clear_flag')
                del invar, invar_cat, outvar
                torch.cuda.empty_cache()
            return total_loss