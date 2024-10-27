import pygrib
import numpy as np
import cv2
from datetime import datetime,timedelta
import torch
from pathlib import Path

try:
    import nvidia.dali as dali
    import nvidia.dali.plugin.pytorch as dali_pth
except ImportError:
    raise ImportError(
        "DALI dataset requires NVIDIA DALI package to be installed. "
        + "The package can be installed at:\n"
        + "https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html"
    )

from data_utils import cos_zenith_angle
from data_utils import latlon_grid
from data_utils import StaticData
import onnx
import onnxruntime as ort


def prepare_input(
    invar,
    cos_zenith=None,
    num_history=0,
    static_data=None,
    step=None,
    time_idx=None,
    stride=1,
    dt=6.0,
    num_samples_per_year=1459,
    device="cuda",
):
    """Prepare input by adding history, cos zenith angle, and static data, if applicable"""

    # Add history
    if num_history > 0:
        # flatten the history dimension
        invar = invar.view(invar.size(0), -1, *(invar.size()[3:]))

    # Add cos zenith
    if cos_zenith is not None:
        cos_zenith = torch.squeeze(cos_zenith, dim=2)
        cos_zenith = torch.clamp(cos_zenith, min=0.0) - 1.0 / np.pi
        invar = torch.concat(
            (invar, cos_zenith[:, step - 1 : num_history + step, ...]), dim=1
        )

    # Add static data
    if static_data is not None:
        invar = torch.concat((invar, static_data), dim=1)

    # Add clock variables
    if time_idx is not None:
        # Precompute the tensors to concatenate
        sin_day_of_year = torch.zeros(1, num_history + 1, 721, 1440, device=device)
        cos_day_of_year = torch.zeros(1, num_history + 1, 721, 1440, device=device)
        sin_time_of_day = torch.zeros(1, num_history + 1, 721, 1440, device=device)
        cos_time_of_day = torch.zeros(1, num_history + 1, 721, 1440, device=device)

        for i in range(num_history + 1):
            # Calculate the adjusted time index
            adjusted_time_idx = (time_idx + i) # time_idx is the time of the year

            # Compute hour of the year and its decomposition into day of year and time of day
            hour_of_year = adjusted_time_idx * stride * dt
            day_of_year = hour_of_year // 24
            time_of_day = hour_of_year % 24

            # Normalize to the range [0, pi/2]
            normalized_day_of_year = torch.tensor(
                (day_of_year / 365) * (np.pi / 2), dtype=torch.float32, device=device
            )
            normalized_time_of_day = torch.tensor(
                (time_of_day / (24 - dt)) * (np.pi / 2),
                dtype=torch.float32,
                device=device,
            )

            # Fill the tensors for the current step
            sin_day_of_year[0, i] = torch.sin(normalized_day_of_year)
            cos_day_of_year[0, i] = torch.cos(normalized_day_of_year)
            sin_time_of_day[0, i] = torch.sin(normalized_time_of_day)
            cos_time_of_day[0, i] = torch.cos(normalized_time_of_day)

        # Concatenate the new channels to invar
        invar = torch.cat(
            (invar, sin_day_of_year, cos_day_of_year, sin_time_of_day, cos_time_of_day),
            dim=1,
        )

    return invar


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


def autogressive_prediction(model=None,invar=None,time_data=None,statict_data=None):
    output = model(invar)

class Timedata_InputIterator(object):
    def __init__(self,year,month,day,hour,num_steps,num_history):
        self.start_time = datetime(year,month,day,hour,0,0)
        self.year=year
        self.num_history=num_history
        
        self.num_steps=num_steps

    def __iter__(self):
        self.i=0
        self.n=self.num_history + self.num_steps + 1
        return self

    def __next__(self):
        
        timestamps = np.array(
            [
                (
                    self.start_time + timedelta(hours=i * 6)
                ).timestamp()
                for i in range(self.num_history + self.num_steps + 1)
            ]
        )
        # time of the year
        time_of_year_idx = (self.start_time-datetime(self.year, 1, 1))/timedelta(hours=6)
        
        return timestamps,np.array([time_of_year_idx])

class PygribDatapipe(object):
    def __init__(self,year,month,day,hour,num_steps,num_history,device):
        
        latlon_bounds = ((90, -90), (0, 360))
        data_latlon = np.stack(
            latlon_grid(bounds=latlon_bounds, shape=[721, 1440]),
            axis=0,
        )
        self.length=1
        self.output_keys = ["cos_zenith", "time_of_year_idx"]
        if num_history == 0:
            self.layout = ["CHW", "FCHW"]
        else:
            self.layout = ["FCHW", "FCHW"]
        timeli = Timedata_InputIterator(year, month, day, hour, num_steps, 1)
        pipe = dali.pipeline.Pipeline(batch_size=1, num_threads=2, device_id=0)
        with pipe:
            timestamps,time_of_year_idx = dali.fn.external_source(
                source=timeli, num_outputs=2,batch=False
            )

            # 计算cos_zenith
            latlon_dali = dali.types.Constant(data_latlon)
            # timestamps_dali = dali.types.Constant(timestamps)
            cos_zenith = dali.fn.cast(
                cos_zenith_angle(timestamps, latlon=latlon_dali),
                dtype=dali.types.FLOAT,
            )
            if device.type == "cuda":
                cos_zenith = cos_zenith.gpu()
                time_of_year_idx = time_of_year_idx.gpu()
            output=(cos_zenith, time_of_year_idx)
            pipe.set_outputs(*output)
            self.pipe = pipe
        
    def __iter__(self):
        # Reset the pipeline before creating an iterator to enable epochs.
        self.pipe.reset()
        # Create DALI PyTorch iterator.
        return dali_pth.DALIGenericIterator([self.pipe], self.output_keys)

    def __len__(self):
        return self.length


def main():
    # constant definition
    year=2023
    month=1
    day=2
    hour=6
    num_steps=60
    
    if torch.cuda.is_available():
        device = torch.device("cuda",0)
    else:
        device = torch.device("cpu")
    
    dtype=torch.float16
    # 输入当前时间t_1,上一时刻t_0
    
    t_1 =  datetime(year,month,day,hour,0,0)
    t_0 =  t_1-timedelta(hours=6)
    # 生成文件路径
    ART_1 = t_1.strftime('/workspaces/cma_data/inference_data/ART_ATM_GLB_0P25_6HOR_ANAL_%Y%m%d%H.grib2')
    LAND_1 =  t_1.strftime('/workspaces/cma_data/inference_data/CRA40LAND_SURFACE_%Y%m%d%H_GLB_0P25_HOUR_V1_0_0.grib')
    ART_0 = t_0.strftime('/workspaces/cma_data/inference_data/ART_ATM_GLB_0P25_6HOR_ANAL_%Y%m%d%H.grib2')
    LAND_0 =  t_0.strftime('/workspaces/cma_data/inference_data/CRA40LAND_SURFACE_%Y%m%d%H_GLB_0P25_HOUR_V1_0_0.grib')
    # 读取ART和LAND数据
    data_1 = grib_read(ART_1,LAND_1)
    data_0 = grib_read(ART_0,LAND_0)

    tensor_1 = torch.from_numpy(data_1)
    tensor_0 = torch.from_numpy(data_0)
    single_shape=list(tensor_1.shape)
    multiple_shape=single_shape
    multiple_shape[0]=num_steps
    # 两个时刻数据合并
    invar=torch.concat((tensor_0,tensor_1),dim=0)
    
    # 标准化
    # load normalisation values
    mean_stat_file = Path("/workspaces/cma_data/stats/model_large_135/global_means.npy")
    std_stat_file = Path("/workspaces/cma_data/stats/model_large_135/global_stds.npy")
    num_channels_climate=135
    channels=[i for i in range(num_channels_climate)]
    # has shape [1, C, 1, 1]
    mu = np.load(str(mean_stat_file))[:, channels]
    # has shape [1, C, 1, 1]
    std = np.load(str(std_stat_file))[:, channels]
    # Normalize using the mean and std
    invar = (invar-mu)/std
    invar=invar.unsqueeze(0).to(device)
    
    static_dataset_path='/workspaces/cma_data/static/'
    latitudes = torch.linspace(-90, 90, steps=721)
    longitudes = torch.linspace(-180, 180, steps=1441)[1:]
    static_data = StaticData(
                static_dataset_path, latitudes, longitudes
            ).get()
    static_data = static_data.to(device)
    datapipe=PygribDatapipe(year, month, day, hour, num_steps, 1, device)
    
    # load model
    onnx_filename='outputs/graphcast/accumodel_large_f16.onnx'
    providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),
                                        "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]
    sess_options = ort.SessionOptions()
    accu_model=ort.InferenceSession(onnx_filename,sess_options=sess_options, providers=providers)

    # pred = (
    #     torch.empty(multiple_shape)
    #     .to(dtype=dtype)
    #     .to(device)
    # )
    pred = np.empty(multiple_shape)
    
    data = next(iter(datapipe))
    cos_zenith=data[0]["cos_zenith"]
    time_idx=data[0]["time_of_year_idx"]
    # autogressive_prediction
    # 自回归预测
    for t in range(num_steps):
        # prepare input
        invar_cat = prepare_input(
                            invar,
                            cos_zenith,
                            num_history=1,
                            static_data=static_data,
                            step=1,
                            time_idx=time_idx,
                            stride=1,
                            dt=6,
                            device=device,
                        )
        invar_cat = invar_cat.to(dtype)
        print(invar_cat.shape)
        invar_cat=invar_cat.detach().cpu().numpy() if invar_cat.requires_grad else invar_cat.cpu().numpy()
        ort_inputs = {accu_model.get_inputs()[0].name:invar_cat}
        outpred = accu_model.run(None, ort_inputs) 
        print('outpred:',outpred[0].shape)
        pred[t] = outpred[0]
        pred_tensor=torch.from_numpy(outpred[0]).to(device)
        # drop the first time step, and append the prediction as the last time step in invar
        invar = torch.cat((invar[:, 1:, :, :], pred_tensor.unsqueeze(1)), dim=1)
        outpred[0]=outpred[0]*std+mu # unnormalize

        #################################################
        # write output file
        import json
        import eccodes

        path='examples/weather/graphcast/inference/cma_data.json'
        with open(path, "r") as f:
            data_json = json.load(f)
            channel_list = data_json["coords"]["channel"]

        cma_chan=['2t', '2sh', '10u', '10v', 'prmsl', 'gh_100', 't_100', 'q_100', 'u_100', 'v_100', 'gh_200', 't_200', 'q_200', 
                'u_200', 'v_200', 'gh_500', 't_500', 'q_500', 'u_500', 'v_500', 'gh_700', 't_700', 'q_700', 'u_700', 'v_700',
                'gh_850', 't_850', 'q_850', 'u_850', 'v_850', 'gh_925', 't_925', 'q_925', 'u_925', 'v_925']#'tp'
        #查找cmachan在channel_list中的索引
        chan_index=[channel_list.index(i) for i in cma_chan]
        chans=0

        # 模板文件和输出文件
        template_file = 'examples/weather/graphcast/inference/output/FDP_OUTPUT_DEMO.grib2'
        LT=(t+1)*6
        output_file1 = f'examples/weather/graphcast/inference/output/CHN_PLEVELS_0P25_HOUR_{year:04d}{month:02d}{day:02d}{hour:02d}_{LT:03d}.grib2'
        output_file2 = f'examples/weather/graphcast/inference/output/CHN_SURFACE_0P25_HOUR_{year:04d}{month:02d}{day:02d}{hour:02d}_{LT:03d}.grib2'
        pressure_variables = ['gh', 't', 'q', 'u', 'v']
        # 打开模板 GRIB2 文件
        with open(template_file, 'rb') as f_in, open(output_file1, 'wb') as f_out1, open(output_file2, 'wb') as f_out2:
            while True:
                if chans==35:
                    break # tp not record
                gid = eccodes.codes_grib_new_from_file(f_in)
                if gid is None:
                    break

                # # 获取 GRIB 消息中的变量名和等压面
                # short_name = eccodes.codes_get(gid, 'shortName')
                # typeOfLevel = eccodes.codes_get(gid, 'typeOfLevel')
                # if typeOfLevel == 'isobaricInhPa':
                #     isobaric_level = eccodes.codes_get(gid, 'level')
                #     cma_chan.append(short_name+'_'+str(isobaric_level))
                # else:
                #     isobaric_level = 0
                #     cma_chan.append(short_name)
                # print(f'Updated channel: {short_name}, level: {isobaric_level} hPa')
                short_name = eccodes.codes_get(gid, 'shortName')

                flattened_values = outpred[0][:,chan_index[chans]].flatten()

                # 将新的数据写入 GRIB 消息
                eccodes.codes_set_values(gid, flattened_values)

                if short_name in pressure_variables:
                    eccodes.codes_write(gid, f_out1)
                else:
                    eccodes.codes_write(gid, f_out2)
                eccodes.codes_release(gid)
                chans=chans+1
                print(chans)
        
        time_idx=time_idx+1 # 0928 add: next time step

    
    
    
    
    
    
if __name__ == '__main__':

    main()
