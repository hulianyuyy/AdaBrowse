# Stage one
This repo holds codes of stage one for the paper: AdaBrowse: Adaptive Video Browser for Efficient Continuous Sign Language Recognition.(ACMMM 2023) [[paper]](https://arxiv.org/abs/2308.08327)

## Prerequisites

- Create a soft link toward sclite: 
  `mkdir ./software`
  `ln -s PATH_TO_KALDI/tools/sctk-2.4.10/bin/sclite ./software/sclite`

## Implementation
Stage one is based on [VAC (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html). Many thanks for their great work!

## Data Preparation
You can choose any one of following datasets to verify the effectiveness of AdaBrowse.

### PHOENIX2014 dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/). Our experiments based on phoenix-2014.v3.tar.gz.

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET/phoenix2014-release ./dataset/phoenix2014`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python dataset_preprocess.py --process-image --multiprocessing
   ```

### PHOENIX2014-T dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET/PHOENIX-2014-T-release-v3/PHOENIX-2014-T ./dataset/phoenix2014-T`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python dataset_preprocess-T.py --process-image --multiprocessing
   ```

### CSL dataset

1. Request the CSL Dataset from this website [[download link]](https://ustc-slr.github.io/openresources/cslr-dataset-2015/index.html)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET ./dataset/CSL`

3. The original image sequence is 1280x720, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python dataset_preprocess-CSL.py --process-image --multiprocessing
   ``` 

### CSL-Daily dataset

1. Request the CSL-Daily Dataset from this website [[download link]](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET ./dataset/CSL-Daily`

3. The original image sequence is 1280x720, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python dataset_preprocess-CSL-Daily.py --process-image --multiprocessing
   ``` 

## Inference
We provide the weights on PHOENIX2014 dataset as an example. 

### PHOENIX2014 dataset

| Resolution | Dev WER  | Test WER  | Pretrained model                                             |
| -------- | ---------- | ----------- | --- |
| 224×224 | 19.7%      | 21.0%       |  [[Baidu]](https://pan.baidu.com/s/1JouGD4lT4nTroodjWWiDtQ) (passwd: 4fr7)<br />[[Google Drive]](https://drive.google.com/file/d/1MK83R-jHEp51UJhmADbR8lJ_QVd_Zx22/view?usp=sharing) | 
| 160×160 | 20.7%      | 21.7%       | [[Baidu]](https://pan.baidu.com/s/1GebF6LqtfrsU12gu6Uv5fQ) (passwd: 689r)<br />[[Google Drive]](https://drive.google.com/file/d/1iwtd6xlQDnokiVBRfsJQ5IZMYCusZEBM/view?usp=sharing) |
| 96×96 | 23.4%      | 23.2%       | [[Baidu]](https://pan.baidu.com/s/1xv2lnMF6DgyBNwO_SmrOAw) (passwd: f344)<br />[[Google Drive]](https://drive.google.com/file/d/1xh0f14UFqR1lClQIh3a5rP3CGKcKuxbQ/view?usp=sharing) |

### Training

The priorities of configuration files are: command line > config file > default values of argparse. To train the SLR model, run the command below:

`python main.py --device your_device`

Note that you can choose the target dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml.

For CSL-Daily dataset, You may choose to reduce the lr by half from 0.0001 to 0.00005, change the lr deacying rate (gamma in the 'optimizer.py') from 0.2 to 0.5, and disable the temporal resampling strategy (comment line 104 in dataloader_video.py).
 