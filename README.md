# EDIS (Energy-Guided Diffusion Sampling for Offline-to-Online Reinforcement Learning)

This repo contains the code of Energy-guided DIffusion Sampling (EDIS) algorithm, proposed by [Energy-Guided Diffusion Sampling for Offline-to-Online Reinforcement Learning](https://openreview.net/attachment?id=hunSEjeCPE&name=pdf#:~:text=To%20address%20this%20issue%2C%20we%20introduce%20an%20innovative,for%20en-hanced%20data%20generation%20in%20the%20online%20phase.). 

EDIS utilizes a diffusion model to extract prior knowledge from the offline dataset and employs energy functions to distill this knowledge for enhanced data generation in the online phase. The generated samples confirm online fine-tuning distribution without oblivion of transition fidelity.

## Getting started

### Install MuJoCo.
- Download [MuJoCo Key](https://www.roboti.us/license.html) and [MuJoCo 2.1 binaries](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz)
- Extract the downloaded `mujoco210` and `mjkey.txt` into `~/.mujoco/mujoco210` and `~/.mujoco/mjkey.txt` 

Add the following environment variables into `~/.bashrc`
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

### Install Anaconda environment.

To install the necessary environments, we use 
```bash
conda create -n edis python=3.8
conda activate edis
pip install -r requirements/requirements_dev.txt
```

### Run the code.

To run the Cal-ql-EDIS or IQL-EDIS, use the command 
```bash
python -u algorithms/iql_edis.py --env hopper-random-v2 --state_guide --policy_guide --transition_guide --seed 48
```
or 
```bash
python -u algorithms/cal_ql_edis.py --env hopper-random-v2 --state_guide --policy_guide --transition_guide --seed 48
```


## Built upon CORL

Our EDIS is built upon CORL, please refer to https://github.com/tinkoff-ai/CORL
