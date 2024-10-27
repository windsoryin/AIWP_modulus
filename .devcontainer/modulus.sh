

# use china image source
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set install.trusted-host https://pypi.tuna.tsinghua.edu.cn

# Update pip and setuptools
pip install "pip==23.2.1" "setuptools==68.2.2"

# Setup git lfs, graphviz gl1(vtk dep)
# apt-get update && \
#     apt-get install -y git-lfs graphviz libgl1 && \
#     git lfs install

export CUDA_COMPAT_TIMEOUT=90


# Install other dependencies
pip install --no-cache-dir "h5py>=3.7.0" "netcdf4>=1.6.3" "ruamel.yaml>=0.17.22" "scikit-learn>=1.0.2" "cftime>=1.6.2" "einops>=0.7.0" "pyspng>=0.1.0"
pip install --no-cache-dir "hydra-core>=1.2.0" "termcolor>=2.1.1" "wandb>=0.13.7" "mlflow>=2.1.1" "pydantic>=1.10.2" "imageio>=2.28.1" "moviepy>=1.0.3" "tqdm>=4.60.0" "gcsfs==2024.2.0"

# copy modulus source
# COPY . /modulus/

# Install Numcodecs (This needs a separate install because Numcodecs ARM pip install has issues)
# A fix is being added here: https://github.com/zarr-developers/numcodecs/pull/315 but the public release is not ready yet.
pip install --no-cache-dir numcodecs; 


# install vtk and pyvista
pip install --no-cache-dir "vtk>=9.2.6"; 

pip install --no-cache-dir "pyvista>=0.40.1"

# Install DGL, below instructions only work for containers with CUDA >= 12.1
# (https://www.dgl.ai/pages/start.html)
export DGL_BACKEND=pytorch
export DGL_BACKEND=$DGL_BACKEND
export DGLBACKEND=$DGL_BACKEND

# TODO: this is a workaround as dgl is not yet shipping arm compatible wheels for CUDA 12.x: https://github.com/NVIDIA/modulus/issues/432
pip install --no-cache-dir --no-deps dgl==2.0.0 -f https://data.dgl.ai/wheels/cu121/repo.html; 


# Install custom onnx
# TODO: Find a fix to eliminate the custom build
# Forcing numpy update to over ride numba 0.56.4 max numpy constraint
# pip install --force-reinstall --no-cache-dir /modulus/deps/onnxruntime_gpu-1.18.0-cp310-cp310-linux_x86_64.whl; \
pip install --no-cache-dir "numpy==1.22.4";


cd /modulus/
pip install -e .[makani] 
pip uninstall nvidia-modulus -y

pip install --no-cache-dir "tensorflow==2.9.0" "warp-lang>=0.6.0"; 

pip install --no-cache-dir "black==22.10.0" "interrogate==1.5.0" "coverage==6.5.0" "protobuf==3.20.3"

# TODO(akamenev): install Makani via direct URL, see comments in pyproject.toml.
pip install --no-cache-dir --no-deps -e git+https://github.com/NVIDIA/modulus-makani.git@v0.1.0#egg=makani
\


# Install torch-scatter, torch-cluster, and pyg


pip install --no-cache-dir "torch_geometric==2.5.3"



cd /modulus/ && pip install .
pip install --no-cache-dir "protobuf==3.20.3"



#

# Install packages for Sphinx build
pip install --no-cache-dir "recommonmark==0.7.1" "sphinx==5.1.1" "sphinx-rtd-theme==1.0.0" "pydocstyle==6.1.1" "nbsphinx==0.8.9" "nbconvert==6.4.3" "jinja2==3.0.3"
wget https://github.com/jgm/pandoc/releases/download/3.1.6.2/pandoc-3.1.6.2-1-amd64.deb && dpkg -i pandoc-3.1.6.2-1-amd64.deb



# ############################################
#extra packages
pip install --no-cache-dir cdsapi h5netcdf
