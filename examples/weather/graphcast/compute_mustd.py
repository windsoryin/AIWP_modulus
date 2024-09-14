import h5py 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

data_dir=Path('/workspaces/cma_data/test')
data_paths = sorted(data_dir.glob("??????.h5")) # ????=yyyy,
data_files = [h5py.File(path, "r") for path in data_paths]
mean_list = []
std_list = []
for idx in range(len(data_files)):
    mean_data = np.zeros((1,240,1,1))
    std_data = np.zeros((1,240,1,1))
    for chans in range(24):
        data = data_files[idx]["fields"][:,chans*10:10*(chans+1),:,:]
        mean_data[:,chans*10:10*(chans+1),:,:] = np.mean(data, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
        std_data[:,chans*10:10*(chans+1),:,:]=np.std(data, axis=(0, 2, 3)).reshape(1, -1, 1, 1)

    np.save('global_stds.npy',std_data)
    np.save('global_means.npy',mean_data)
    # 将均值和标准差存入列表
    mean_list.append(mean_data.reshape(1, -1, 1, 1))
    std_list.append(std_data.reshape(1, -1, 1, 1))
    np.save('global_stds_'+str(idx)+'.npy',std_list)
    np.save('global_means_'+str(idx)+'.npy',mean_list)
# std=np.load('/workspaces/cma_data/stats/global_stds.npy')

# figure plot
# for idx in range(len(data_files)):
#     for n in range(10):
#         data = data_files[idx]["fields"][4*n:4*(n+1),:6,:,:]
#         mean_data = np.mean(data, axis=(0, 2, 3))
#         std_data=np.std(data, axis=(0, 2, 3))
#         # 将均值和标准差存入列表
#         mean_list.append(mean_data.reshape(1, -1, 1, 1))
#         std_list.append(std_data.reshape(1, -1, 1, 1))
#         s=np.array(std_list)
#         # np.save('global_stds_'+str(idx)+'.npy',std_data.reshape(1, -1, 1, 1))
#         # np.save('global_means_'+str(idx)+'.npy',mean_data.reshape(1, -1, 1, 1))
#         fig, ax = plt.subplots(2, data.shape[0], figsize=(15, 5))
#         fig.subplots_adjust(hspace=0.5, wspace=0.3)
#         channels=[0,2]
#         for c_i,chan in enumerate(channels):
#             for t in range(data.shape[0]):
#                 im_pred = ax[c_i, t].imshow(data[t, chan])
#                 ax[c_i, t].set_title(f"Prediction (t={t+1})", fontsize=10)
#                 fig.colorbar(
#                     im_pred, ax=ax[c_i, t], orientation="horizontal", pad=0.4
#                 )
#         fig.savefig('val_png/val_'+str(idx)+'_'+str(n)+'.png'
#                     )
print(idx)


