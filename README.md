# Rain Streak Removal

### Directory Structure
```
Rain_Streak_Removal/                                     # project directory
├── dataset/ 
│   ├── test/                                            # testing set
│   ├── train/                                           # training set
│   └── val/                                             # validation set
├── MPRNet/                                              # code
├── pytorch-gradual-warmup-lr/                           # warmup scheduler package
├── model/                                               # save the model
├── evaluate_result/                                     # evaluation result of model
├── Hyperparam_Tuning/                                   # Hyperparam tuning experiment(code and result)
├── Ablation_Result/                                     # Ablation experiment result
└── README.md                                            # this file 
```

### Dataset
The dataset is an open-source dataset obtained from the Internet.  
Available link(Google Drive):  
Train dataset `https://drive.google.com/drive/folders/1Hnnlc5kI0v9_BtfMytC2LR5VpLAFZtVe?usp=sharing`    
Test dataset `https://drive.google.com/drive/folders/1PDWggNh8ylevFmrjo-JEvlmqsDlWWvZs?usp=sharing`  
Real rain test dataset `https://drive.google.com/drive/folders/1rk7jdBZifNe_OKJ6j-0Ne8gjYYIg6Qc5`  
The validation dataset is 20% of the images segmented from the training dataset
The dataset structure should be like this:
```
dataset/ 
├── test/                                                # testing dataset
│   └── test_datasets/                                   # like training dataset strcture
├── train/                                               # training dataset
│   ├── norain/                                          # original images
│   └── rain/                                            # rain streak images
└── val/                                                 # validation dataset
```

### Environment
Create new environment
```
conda create -n Rain python=3.9 -y
conda activate Rain
```
Install core packages
```
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install opencv scikit-learn matplotlib -c conda-forge -y
conda install natsort tqdm yacs -c conda-forge -y
conda install scikit-image pandas -c conda-forge -y
```
Install warmup scheduler
```
cd pytorch-gradual-warmup-lr
python setup.py install
```

### Training Model
Under the 'MPRNet' directory
```
python model_train.py
```

### Testing Model
Under the 'MPRNet' directory
```
python model_test.py
```

### Evaluate Result
Under the 'MPRNet' directory
```
python model_evaluate.py
```

