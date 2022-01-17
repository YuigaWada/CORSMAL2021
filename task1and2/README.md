# corsmal_challenge
repo. for [**The CORSMAL challenge: Audio-visual object classification for human-robot collaboration**](https://corsmal.eecs.qmul.ac.uk/challenge.html)

### Required env.
> Hardware
> - CentOS Linux release 7.7.1908 (server machine)
> - Kernel: 3.10.0-1062.el7.x86_64
> - GPU: (4) GeForce GTX 1080 Ti
> - GPU RAM: 48 GB
> - CPU: (2) Xeon(R) Silver 4112 @ 2.60GHz
> - RAM: 64 GB
> - Cores: 24
> 
> Libraries
> - Anaconda 3 (conda 4.7.12)
> - CUDA 7-10.2
> - Miniconda 4.7.12

#### additional information
- The version of python3 in miniconda4.7.12 (pyenv) is `python3.7.4`

#### developer's env.
Hardware
- Ubuntu 20.04LTS
- GPU: GeForce RTX 3080
- GPU RAM: 16GB GDDR6
- CPU: Core i9 11980HK
- RAM: 64 GB
Libraries
- CUDA 11.1
- Miniconda 4.7.12 (pyenv)

### Dataset
Download from [official page](https://corsmal.eecs.qmul.ac.uk/containers_manip.html) & unzip them into `data/` directory.  
See `data/` directory for more information.

## Get Start
### Installation
1. install `torch`, `torchaudio` by pip.. See https://pytorch.org/ and check your hardware requirements
2. In this directory,  
    `python -m pip install ./ --use-feature=in-tree-build`
3. additionally, if you want to rewrite the files in `corsmal_challenge`, remove it from dependencies
    `python -m pip uninstall corsmal_challenge`
