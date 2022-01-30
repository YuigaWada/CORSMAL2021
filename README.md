# Corsmal Challenge 2021
repo. for [**The CORSMAL challenge: Audio-visual object classification for human-robot collaboration**](https://corsmal.eecs.qmul.ac.uk/challenge.html)

<!-- ## Get Start -->
## Installation
### Requirements

- python=3.7.11
- numpy=1.21.2
- scipy=1.6.2
- pandas=1.3.5
- matplotlib=3.5.0
- opencv-python=4.5.5.62

- torch=1.8.2
- torchaudio=0.8.2
- torchvision=0.9.2

### Instructions

0. Clone repository. `git clone https://github.com/YuigaWada/CORSMAL2021.git`

1. Create a new conda environment and activate it. `conda env create -f env.yml && conda activate corsmal`

2. The above command may fail depending on your environment. If it fails, you need to maually install some libraries.
    1. If you do not have `torch`, `torchaudio` or `torchvision` installed, install them by pip.. See https://pytorch.org/ and check your hardware requirements.
    2. To install `pandas`, run
    `conda install pandas` or `conda install pandas=1.3.5=py37h8c16a72_0`
    3. To install `scipy`, run
    `conda install scipy` or `conda install scipy=1.6.2=py37had2a1c9_1`
    4. To install `matplotlib`, run
    `conda install matplotlib` or `conda install matplotlib=3.5.0=py37h06a4308_0`
    5. To install `opencv-python`, run
    `conda install opencv-python` or `pip install opencv-python==4.5.5.62`

## Test
To output the estimations as a `.csv` file, run the following command:
```
python run.py [test_data_path] [output_path] -m12 [task1and2_model] -m4 [task4_model]
```
For example,
```
python run.py ./test ./output.csv -m12 task1and2.pt -m4 task4.pt
```
The pre-trained models are available [here](https://drive.google.com/drive/folders/1QIs-POJIBtgDl1ufYrX5Sopf6RsHjKSb).

The directory of test data should be organized as:

```
test
|-----audio
|-----view1
|     |-----rgb
|     |-----calib
|-----view2
|     |-----rgb
|     |-----calib
|-----view3
|     |-----rgb
|     |-----calib
```
<!-- Run `inference.py` -->

## Training
To train the model, organize your training data as:

```
train
|-----audio
|-----ccm_train_annotation.json
```

### Task 1 and 2

```
python train.py [train_data_path] --task1and2
```
For example,
```
python train.py ./train --task1and2
```

<!-- Place yourself at `CORSMAL2021/task1and2`.
To train the model of Task 1 and 2, organize your training data as:
```
train
|-----audio
```
and put the directory in `CORSMAL2021/task1and2/data/`. 

To start the training, run
```
python ./experiments/20220120-training-2.py
``` -->

### Task 4

```
python train.py [train_data_path] --task4
```
For example,
```
python train.py ./train --task4
```

### Task 3 and 5
There is no training necessary for Task 3 and 5: the algorithms used in Task 3 and 5 are not deep learning ones.


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
We installed pytorch for `CUDA 11.1` with pip.

- torch==1.8.2+cu111
- torchaudio==0.8.2
- torchvision==0.9.2+cu111


<!-- ## Dataset
Download from [official page](https://corsmal.eecs.qmul.ac.uk/containers_manip.html) & unzip them into `data/` directory.  
See `data/` directory for more information. -->
