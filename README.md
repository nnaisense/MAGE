# MAGE :crystal_ball:

<img src="mage_logo.png" width="220" height="220">

The authors' implementation of the MAGE algorithm from ["How to Learn a Useful Critic? Model-based Action-Gradient-Estimator Policy Optimization"](https://arxiv.org/abs/2004.14309).

## Cite as

> Pierluca D'Oro, Wojciech Jaśkowski. "How to Learn a Useful Critic? Model-based Action-Gradient-Estimator Policy Optimization". In: _NeurIPS_, 2020.

```bibtex
@inproceedings{doro2020howto,
    title={How to Learn a Useful Critic? Model-based Action-Gradient-Estimator Policy Optimization},
    author={D'Oro, Pierluca and Ja{\'s}kowski, Wojciech},
    booktitle={NeurIPS},
    year={2020},
  }
```

## Install

1. You should already have them, but just in case, install the libs:

    ```bash
    sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
    ```

1. Create conda environment with the required dependencies:

    ```bash
    conda env create -f conda_env.yml
    ```

1. Download and setup MuJoCo 1.50:

    ```bash
    mkdir ~/.mujoco/
    cd .mujoco/
    wget -c https://www.roboti.us/download/mujoco150_linux.zip
    unzip mjpro150_linux.zip
    rm mjpro150_linux.zip
    ```

    Obtain MuJoCo license key and place it `.mujoco/` directory created above with filename `mjkey.txt`.

    Append the following to `~/.bashrc`:

    ```bash
    # MuJoCo
    if [ -f /usr/lib/x86_64-linux-gnu/libGLEW.so ]; then
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco150/bin:/usr/lib/nvidia-390:/usr/lib/nvidia-375
        export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
    fi

    ```

    Test the MuJoCo installation:

    ```python
    >>> import gym
    >>> gym.make('HalfCheetah-v2')
    ```

1. (Optional) Create a [neptune.ai](neptune.ai) account for logging. Setup your Neptune:

```bash
export NEPTUNE_API_TOKEN=<your neptune.ai token>
```

### Running

 MAGE-TD3 (ours):

```bash
python main.py with env_name=GYMMB_HalfCheetah-v2 agent_alg=td3 tdg_error_weight=5. td_error_weight=1. neptune_project=<optionally_your_neptune_project_name>
```

(Note: `tdg_error_weight=5` corresponds to `lambda=0.2` in the paper)

Dyna-TD3 (model-based baseline):

```bash
python main.py with env_name=GYMMB_HalfCheetah-v2 agent_alg=td3 tdg_error_weight=0. td_error_weight=1. neptune_project=<optionally_your_neptune_project_name>
```
