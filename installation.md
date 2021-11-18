# Installation Instructions

Please follow the instructions below to install the package.

**Primary Dependency: Ubuntu 18.04**

- Clone the repository:
```bash
cd ~
git clone https://github.com/UtkarshMishra04/bioimitation-gym
```

- Install Opensim Dependenices:
```bash
cd bioimitation-gym
bash scripts/build_opensim-core
```
This will install opensim-core in the `opensim-core` directory and will install the dependencies for the package in the `opensim-core/install` directory. Please give some time for the installation to complete.

- Configure Opensim-core to use the correct opensim-core directory by appending the following line to the `~/.bashrc` file:
```bash
export OPENSIM_HOME=~/path/to/bioimitation-gym/opensim-core/install
export OpenSim_DIR=$OPENSIM_HOME/lib/cmake/OpenSim
export LD_LIBRARY_PATH=$OPENSIM_HOME/lib:$LD_LIBRARY_PATH
export PATH=$OPENSIM_HOME/bin:$PATH
export PATH=$OPENSIM_HOME/libexec/simbody:$PATH
```

- Load the environment variables by running the following command:
```bash
source ~/.bashrc
```

- Create a virtual environment by running the following command:
```bash
cd ~
virtualenv bioimitation --python=python3.6
```
Please use the python version specified in the above command. The code and opensim installation is not compatible with other versions of python.

- Activate the virtual environment by running the following command:
```bash
source ~/bioimitation/bin/activate
```

- Install opensim in the virtual environment by running the following command:
```bash
cd $OPENSIM_HOME/lib/python3.6/site-packages
python setup.py install
```
This will install `opensim==4.1` in the virtual environment. Now you are ready to install the `bioimitation-gym` package.

- Install the package by running the following command:
```bash
cd ~/biomitation-gym
pip install -e .
```
This will automatically read and install all dependencies from `scripts/requirements.txt`.