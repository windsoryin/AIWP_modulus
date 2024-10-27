
import pygrib
import numpy as np
import cv2
from datetime import datetime,timedelta

def griblist_to_arraydata(grb_list):
    # 
    if not grb_list:
        raise ValueError("列表不能为空")
    f_array = grb_list[0].values
    array_shape = f_array.shape
    num_arrays = len(grb_list)

    # #(channel,721,1440)
    three_d_array = np.empty((num_arrays, *array_shape), dtype='float32')

    # grib_message2array
    for i, item in enumerate(grb_list):
        # 高空数据纬度上721，直接加入，地面数据720需要先插值
        if(item.values.shape[0]==721):
            three_d_array[i] = item.values
        else:
            # 使用 OpenCV 的 resize 函数进行双线性插值
            new_shape = (1440, 721)  # OpenCV 的 resize 函数需要 (宽, 高) 顺序
            interpolated_data = cv2.resize(item.values, new_shape, interpolation=cv2.INTER_LINEAR)
            three_d_array[i] = interpolated_data

    #(1,channel,721,1440)
    three_d_array = np.expand_dims(three_d_array,axis=0)
    return three_d_array


def grib_read(ART,LAND):
    # grib文件读取
    grbs_ART = pygrib.open(ART)
    grbs = grbs_ART(shortName=['gh', 't', 'q', 'u', 'v','prmsl'],typeOfLevel=['isobaricInhPa','meanSea'])
    grbs_LAND = pygrib.open(LAND)
    grbs.extend(grbs_LAND(typeOfLevel=['heightAboveGround']))
    # 数据通道筛选
    index = [5, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    new_list = [list(range(i * 5, (i + 1) * 5)) for i in index]
    flattened_channels = [item for sublist in new_list for item in sublist]
    grbs_silced = [grbs[i] for i in flattened_channels]
    # 大气、地表数据合成
    grbs_data = griblist_to_arraydata(grbs_silced)
    return grbs_data

file='examples/weather/graphcast/inference/output/CHN_SURFACE_0P25_HOUR_2023010206_006.grib2'
grbs_ART = pygrib.open(file)
grbs = grbs_ART(shortName=['2t', '2sh', '10u', '10v', 'prmsl',])

import xarray as xr
ds = xr.open_dataset('examples/weather/graphcast/inference/output/FDP_OUTPUT_DEMO.grib2', engine='cfgrib')


import numpy as np
from eccodes import *

# 模板文件和输出文件
template_file = 'examples/weather/graphcast/inference/output/FDP_OUTPUT_DEMO.grib2'
output_file = 'examples/weather/graphcast/inference/output/output.grib2'


# 假设我们有一个三维数组数据，形状为 [36, 721, 1440]
# 其中，36 是通道数，721 是纬度点数，1440 是经度点数
new_data = np.random.rand(36, 721, 1440)

# 通道名称列表，长度为 36，与 new_data 的第一个维度对应
import json
path='examples/weather/graphcast/inference/cma_data.json'
with open(path, "r") as f:
    data_json = json.load(f)
    channel_list = data_json["coords"]["channel"]

cma_chan=['2t', '2sh', '10u', '10v', 'prmsl', 'gh_100', 't_100', 'q_100', 'u_100', 'v_100', 'gh_200', 't_200', 'q_200', 
           'u_200', 'v_200', 'gh_500', 't_500', 'q_500', 'u_500', 'v_500', 'gh_700', 't_700', 'q_700', 'u_700', 'v_700',
           'gh_850', 't_850', 'q_850', 'u_850', 'v_850', 'gh_925', 't_925', 'q_925', 'u_925', 'v_925']#'tp'

#查找cmachan在channel_list中的索引
index=[channel_list.index(i) for i in cma_chan]

# 对应通道的等压面列表（假设每个通道可能对应不同的等压面）
# 每个通道有一个等压面值（单位：hPa）
isobaric_levels = [1000, 850, 700, 500, 300, 200, ...]

# 打开模板 GRIB2 文件
with open(template_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
    while True:
        gid = codes_grib_new_from_file(f_in)
        if gid is None:
            break

        # 获取 GRIB 消息中的变量名和等压面
        short_name = codes_get(gid, 'shortName')
        typeOfLevel = codes_get(gid, 'typeOfLevel')
        if typeOfLevel == 'isobaricInhPa':
            isobaric_level = codes_get(gid, 'level')
            cma_chan.append(short_name+'_'+str(isobaric_level))
        else:
            isobaric_level = 0
            cma_chan.append(short_name)
        # # 查找与当前消息匹配的通道和等压面
        # for i, channel_name in enumerate(channel_names):
        #     if short_name == channel_name and isobaric_level == isobaric_levels[i]:
        #         # 找到匹配的通道和等压面
        #         channel_index = i

        #         # 提取新数据中对应通道的数据 (纬度 x 经度)
        #         channel_data = new_data[channel_index, :, :]

        #         # 确保一维化数据后可以写回 GRIB
        #         flattened_values = channel_data.flatten()

        #         # 将新的数据写入 GRIB 消息
        #         codes_set_values(gid, flattened_values)

        #         # 打印匹配的消息和通道，以便调试
        #         print(f'Updated channel: {short_name}, level: {isobaric_level} hPa')

        # # 将修改后的消息写入新文件
        # codes_write(gid, f_out)

        # 释放 GRIB 资源
        codes_release(gid)

print(f'新文件保存为 {output_file}')
print(cma_chan)