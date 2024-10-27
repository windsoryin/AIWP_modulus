import pygrib
import numpy as np
import cv2
from datetime import datetime,timedelta
import torch
from pathlib import Path

import onnx
import onnxruntime as ort
import configparser
import os
import sys
import time

import json
import eccodes
import time

from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def get_lat_lon(var1,region_mask,lats,lons):
    # 获取该区域内的值
    region_values = np.ma.masked_where(~region_mask, var1)
    # plot region_values
    # 找到区域内的最小值及其索引
    min_value = np.min(region_values)
    min_index = np.argmin(region_values)

    # 根据最小值的索引找到对应的经纬度
    min_lat_value = lats.flat[min_index]
    min_lon_value = lons.flat[min_index]
    
    # 以最小值附近的3x3网格区域为例，找到附近的网格点
    grid_size = 3  # 调整这个大小来扩大或缩小插值区域

    lat_range_mask = (lats >= min_lat_value - grid_size) & (lats <= min_lat_value + grid_size)
    lon_range_mask = (lons >= min_lon_value - grid_size) & (lons <= min_lon_value + grid_size)
    local_region_mask = lat_range_mask & lon_range_mask

    # 获取局部区域的值和对应的经纬度
    local_values = var1[local_region_mask]
    local_lats = lats[local_region_mask]
    local_lons = lons[local_region_mask]

    # 创建插值网格，生成更细的局部网格
    num_points = 20  # 插值点数量，可以根据需要调整
    local_grid_lat = np.linspace(min_lat_value - grid_size, min_lat_value + grid_size, num_points)
    local_grid_lon = np.linspace(min_lon_value - grid_size, min_lon_value + grid_size, num_points)
    local_grid_lon, local_grid_lat = np.meshgrid(local_grid_lon, local_grid_lat)

    # 使用双线性插值方法进行局部插值
    interpolated_local_values = griddata(
        points=(local_lats, local_lons),
        values=local_values,
        xi=(local_grid_lat, local_grid_lon),
        method='linear'
    )

    # 找到局部插值数据中的最小值及其对应经纬度
    min_local_value = np.min(interpolated_local_values)
    min_local_index = np.argmin(interpolated_local_values)
    min_local_lat_value = local_grid_lat.flat[min_local_index]
    min_local_lon_value = local_grid_lon.flat[min_local_index]
    
    # min_local_value=min_value
    # min_local_lat_value=min_lat_value
    # min_local_lon_value=min_lon_value
    return min_local_value,min_local_lat_value,min_local_lon_value  

def main(year, month, day, hour):
    cfg = configparser.ConfigParser()
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(curr_dir, 'config.ini')
    cfg.read(file_path)
    result_dir = os.path.join(cfg.get("path",'ResultDir'))

    # constant definition
    # year=cfg.getint("info",'year')
    # month=cfg.getint("info",'month')
    # day=cfg.getint("info",'day')
    # hour=cfg.getint("info",'hour')
    num_steps=cfg.getint("info",'num_steps')
    
    if torch.cuda.is_available():
        device = torch.device("cuda",0)
    else:
        device = torch.device("cpu")
    
    dtype=torch.bfloat16
    # 输入当前时间t_1,上一时刻t_0
    

    # 地表数据
    variable_land = ['2t','2sh','10u','10v','prmsl']
    # 示例：提取 't' shortName (假设是温度) 在 850 hPa 高度的变量
    pressure_variables = ['t', 'gh', 'q', 'u', 'v']
    pressure_levels=[100,200,500,700,850,925] # unit:hpa
    # 创建一个存储所有变量差异的字典
    variable_bias = np.empty((num_steps,len(pressure_variables),len(pressure_levels)),dtype='float32')
    variable_rmse = np.empty((num_steps,len(pressure_variables),len(pressure_levels)),dtype='float32')
    variable_bias_land = np.empty((num_steps,len(variable_land)),dtype='float32')
    variable_rmse_land = np.empty((num_steps,len(variable_land)),dtype='float32')
    for t in range(num_steps):
        LT=(t+1)*6
        # 生成文件路径
        t_0 =  datetime(year,month,day,hour,0,0) # 起报时间
        t_1 =  datetime(year,month,day,hour,0,0)+(t+1)*timedelta(hours=6)
        ART_true = t_1.strftime(os.path.join(cfg.get("path",'TruedataDir'), f'%Y/%Y%m%d/ART_ATM_GLB_0P25_6HOR_ANAL_%Y%m%d%H.grib2'))
        LAND_true =  t_1.strftime(os.path.join(cfg.get("path",'TruedataDir'), f'%Y/CRA40LAND_SURFACE_%Y%m%d%H_GLB_0P25_HOUR_V1_0_0.grib'))
        ART_pred = t_0.strftime(os.path.join(cfg.get("path",'PreddataDir'), f'%Y/%Y%m%d/GLB_PLEVELS_0P25_HOUR_%Y%m%d%H_{LT:03d}.grib2'))
        LAND_pred =  t_0.strftime(os.path.join(cfg.get("path",'PreddataDir'), f'%Y/%Y%m%d/GLB_SURFACE_0P25_HOUR_%Y%m%d%H_{LT:03d}.grib2'))
        
        
        grbs_ART_true = pygrib.open(ART_true)
        grbs_ART_pred = pygrib.open(ART_pred)
        grbs_ART_true = grbs_ART_true(shortName=['gh', 't', 'q', 'u', 'v'],typeOfLevel=['isobaricInhPa'],level=[100,200,500,700,850,925])
        cma_chan=[]
        for grib in grbs_ART_true:
            short_name=grib.shortName
            isobaric_level=grib.level
            cma_chan.append(short_name+'_'+str(isobaric_level))
        cma_chan2=[]
        for grib in grbs_ART_pred:
            short_name=grib.shortName
            isobaric_level=grib.level
            cma_chan2.append(short_name+'_'+str(isobaric_level))
        
        
        for n in range(len(grbs_ART_true)):
            var1 = grbs_ART_true[n].values
            var2 = grbs_ART_pred[n+1].values
            difference = var1 - var2
            rmse=np.sqrt(np.mean(difference**2))
            i= pressure_variables.index(grbs_ART_true[n].shortName)
            j= pressure_levels.index(grbs_ART_true[n].level)
            variable_bias[t, i, j] = np.mean(difference)
            variable_rmse[t, i, j] = rmse
        
        grbs_ART_true=pygrib.open(ART_true)
        grbs_true=grbs_ART_true(shortName=['prmsl'])
        grbs_LAND_true = pygrib.open(LAND_true)
        grbs_LAND_true = grbs_LAND_true(typeOfLevel=['heightAboveGround'])
        grbs_LAND_true.extend(grbs_true)
        grbs_LAND_pred = pygrib.open(LAND_pred)



        for i in range(len(grbs_LAND_true)):
            var1=grbs_LAND_true[i].values
            var2=grbs_LAND_pred[i+1].values
            new_shape = (1440, 721)  # OpenCV 的 resize 函数需要 (宽, 高) 顺序
            var1 = cv2.resize(var1, new_shape, interpolation=cv2.INTER_LINEAR)
            
            lats, lons = grbs_LAND_pred[i+1].latlons()  # 获取经纬度数组
            # 给定的经纬度范围 (min_lat, max_lat, min_lon, max_lon)
            min_lat, max_lat = 10, 30  # 纬度范围
            min_lon, max_lon = 110, 140  # 经度范围
            # if i==4:
            #     # 创建掩膜，筛选给定区域的经纬度
            #     lat_mask = (lats >= min_lat) & (lats <= max_lat)
            #     lon_mask = (lons >= min_lon) & (lons <= max_lon)
            #     region_mask = lat_mask & lon_mask
                
            #     # 获取非掩膜区域的经纬度序号
            #     lat_indices, lon_indices = np.where(region_mask)
            #     extent = [lon_indices.min(), lon_indices.max(), lat_indices.min(), lat_indices.max()]
            #     region_values_resized_var1 = var1[extent[2]:extent[3]+1, extent[0]:extent[1]+1]
            #     region_values_resized_var2 = var2[extent[2]:extent[3]+1, extent[0]:extent[1]+1]

            #     # plt.figure(figsize=(12, 6))

            #     # plt.subplot(1, 2, 1)
            #     # plt.imshow(region_values_resized_var1)
            #     # plt.title('Region Values - var1')
            #     # plt.xlabel('Longitude')
            #     # plt.ylabel('Latitude')
            #     # plt.colorbar()

            #     # plt.subplot(1, 2, 2)
            #     # plt.imshow(region_values_resized_var2)
            #     # plt.title('Region Values - var2')
            #     # plt.xlabel('Longitude')
            #     # plt.ylabel('Latitude')
            #     # plt.colorbar()

            #     # plt.savefig('/workspaces/AIWP_modulus/examples/weather/graphcast/inference/png/region_values_comparison.png')

            #     min_value, min_lat, min_lon = get_lat_lon(var1, region_mask, lats, lons)
            #     min_value_pred,min_lat_pred,min_lon_pred = get_lat_lon(var2,region_mask,lats,lons)

            #     # save the result in csv file
            #     with open(os.path.join(result_dir,'track.csv'), 'a') as f:
            #         f.write(t_1.strftime('%Y-%m-%d %H:%M:%S') + f"{min_value},{min_lat},{min_lon},"+f"{min_value_pred},{min_lat_pred},{min_lon_pred}"+'\n')
            #         #f.write(f"{min_value},{min_lat},{min_lon}\n")
            #         # f.write(f"{min_value_pred},{min_lat_pred},{min_lon_pred}\n")

            difference = var1 - var2
            rmse=np.sqrt(np.mean(difference**2))
            variable_bias_land[t,i] = np.mean(difference)
            variable_rmse_land[t,i] = rmse
        
        print(t)
        
    # svae the result
    np.save(os.path.join(result_dir, f'result_new_40/variable_bias.npy'), variable_bias)
    np.save(os.path.join(result_dir, f'result_new_40/variable_bias_land.npy'), variable_bias_land)
    np.save(os.path.join(result_dir, f'result_new_40/variable_rmse_land.npy'), variable_rmse_land)
    np.save(os.path.join(result_dir, f'result_new_40/variable_rmse.npy'), variable_rmse)


def plot_rmse():
    cfg = configparser.ConfigParser()
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(curr_dir, 'config.ini')
    cfg.read(file_path)
    result_dir = os.path.join(cfg.get("path",'ResultDir'))
    variable_rmse=np.load(os.path.join(result_dir, f'result_40/variable_rmse.npy'))
    variable_rmse2=np.load(os.path.join(result_dir, f'result_new_40/variable_rmse.npy'))

     # 地表数据
    variable_land = ['2t','2sh','10u','10v','prmsl']
    # 示例：提取 't' shortName (假设是温度) 在 850 hPa 高度的变量
    pressure_variables = ['t', 'gh', 'q', 'u', 'v']
    pressure_levels=[100,200,500,700,850,925] # unit:hpa
    plt.figure(figsize=(15, 10))
    # subplot the rmse of pressure variables at different levels along the time steps x axis
    for i in range(5):
        for j in range(6):
            if i==0:
                variable_rmse[:,i,j]=variable_rmse[:,i,j]/4
                variable_rmse2[:,i,j]=variable_rmse2[:,i,j]/4
            if i>2:
                variable_rmse[:,i,j]=variable_rmse[:,i,j]/3
                variable_rmse2[:,i,j]=variable_rmse2[:,i,j]/3
                        
            ax = plt.subplot(5, 6, i*6+j+1)
            ax.plot(variable_rmse[:,i,j])
            ax.plot(variable_rmse2[:,i,j])
           
            ax.set_title(f'{pressure_variables[i]}-{pressure_levels[j]} hPa', fontsize=8)
            if i == 4:
                ax.set_xlabel('Time Steps')
                if j==4:
                    ax.legend(['AccuModel','New Model'])
            if j == 0:
                ax.set_ylabel('RMSE')
            ax.tick_params(axis='both', which='major', labelsize=6)
    # save the result as png
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'png/variable_rmse.png'))

if __name__ == '__main__':
    
    main(2023, 7, 23, 6)
    plot_rmse()