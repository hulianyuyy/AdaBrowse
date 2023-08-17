# AdaBrowse
This repo holds codes of the paper: AdaBrowse: Adaptive Video Browser for Efficient Continuous Sign Language Recognition.(ACMMM 2023) [[paper]](https://arxiv.org/abs/2308.08327)

This repo is based on [VAC (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html). Many thanks for their great work!

## Prerequisites

- This project is implemented in Pytorch (>1.8). Thus please install Pytorch first.

- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.

- sclite [[kaldi-asr/kaldi]](https://github.com/kaldi-asr/kaldi), install kaldi tool to get sclite for evaluation. 

- [SeanNaren/warp-ctc](https://github.com/SeanNaren/warp-ctc) for ctc supervision.

## Implementation
We now implement our AdaBrowse with three resolution candidates: {96×96, 160×160, 224×224}, and three subsequence lengths: {1/4, 1/2, 1.0}.

## Data Preparation
You can choose any one of following datasets to verify the effectiveness of AdaBrowse.

### PHOENIX2014 dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/). Our experiments based on phoenix-2014.v3.tar.gz.


### PHOENIX2014-T dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)


### CSL dataset

1. Request the CSL Dataset from this website [[download link]](https://ustc-slr.github.io/openresources/cslr-dataset-2015/index.html)

### CSL-Daily dataset

1. Request the CSL-Daily Dataset from this website [[download link]](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/)


## Inference

Due to some practical reasons for system deployment, we only provide the weights of stage one and now don't release the weights of stage two for AdaBrowse. One can train the model from stage one to verify the effectiveness of AdaBrowse.

### Training

First, follow the instructions of Stage_one to prepare the weights for resolutions of {96×96, 160×160, 224×224}, or directly use the weights provided by us.

Second, follow the instructions of Stage_two to train AdaBrowse.
 