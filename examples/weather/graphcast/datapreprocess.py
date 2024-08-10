import h5py
import numpy as np
import netCDF4 as nc

def get_metadata(file):
    # Add your implementation here to retrieve metadata from the file
    pass


aa=nc.Dataset('/workspaces/AIWP_modulus/examples/weather/graphcast/geopotential.nc')["z"][:]
# data2=np.load('/workspaces/AIWP_modulus/examples/weather/graphcast/2015.npy')

# data=np.load('/workspaces/data/static/orography.npy')

# Open the HDF5 file
file_path = '/workspaces/data/static/orography.h5'
with h5py.File(file_path, 'r') as file:
    dataset_names = list(file.keys())
    ss=file["orog"][:]
        # print(file[key].value)
    metadata = get_metadata(file)
    print(1)
    # json.dump(metadata, f)
    # Access the dataset or attributes within the file
    # Example: dataset = file['dataset_name']
    # Example: attribute = file.attrs['attribute_name']
    
    # Add your code here to process the data from the HDF5 file
    # ...


