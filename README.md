# Corsmal Challenge 2021
repo. for [**The CORSMAL challenge: Audio-visual object classification for human-robot collaboration**](https://corsmal.eecs.qmul.ac.uk/challenge.html)

## Get Start
### Installation
1. Run 
`conda env create -f env.yml`
2. The above command may fail depending on your environment. If it fails, you need to maually install some libraries.
    1. If you do not have `torch` or `torchaudio` installed, install them by pip.. See https://pytorch.org/ and check your hardware requirements.
    2. To install `pandas`, run
    `conda install pandas`
    3. To install `opencv`, run
    `conda install opencv`
    4. To install `scipy`, run
    `conda install -c anaconda scipy`
3. If you want to train models, you also need to run
`conda install numpy ipykernel matplotlib torchinfo -c conda-forge`
<!-- 2. In this directory,  
    `python -m pip install ./ --use-feature=in-tree-build`
3. Additionally, if you want to rewrite the files in `corsmal_challenge`, remove it from dependencies
    `python -m pip uninstall corsmal_challenge` -->

### Test
To output the estimations as a `.csv` file, run the following command:
```
python run.py [test_data_path] [output_path] -m12 [task1and2_model] -m4 [task4_model]
```
For example,
```
python run.py ./test ./output -m12 task1and2.pt -m4 task4.pt
```
The pre-trained models are available [here](https://drive.google.com/drive/folders/1QIs-POJIBtgDl1ufYrX5Sopf6RsHjKSb).

<!-- Run `inference.py` -->

## Environment settings
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

### Developer's env.
Hardware
- Ubuntu 20.04LTS
- GPU: GeForce RTX 3080 laptop
- GPU RAM: 16GB GDDR6
- CPU: Core i9 11980HK
- RAM: 64 GB
Libraries
- CUDA 11.1
- Miniconda 4.7.12 (pyenv)

### Additional information
The version of python3 in miniconda4.7.12 (pyenv) is `python3.7.4`

<!-- ## Dataset
Download from [official page](https://corsmal.eecs.qmul.ac.uk/containers_manip.html) & unzip them into `data/` directory.  
See `data/` directory for more information. -->
