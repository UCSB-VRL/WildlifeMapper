<img src="./docs/thumbnail_wildlifemapper2.png" width="200">

## WildlifeMapper: Aerial Image Analysis for Multi-Species Detection and Identification

WildlifeMapper (WM) is a state-of-the-art model for detecting, locating, and identifying multiple animal species in aerial imagery. It introduces novel modules to enhance localization and identification accuracy, with a verified dataset of 11k images and 28k annotations. This repository contains code for WildlifeMapper, scripts to download and tool to visualize dataset ([**BisQue**](https://bisque2.ece.ucsb.edu/client_service/view?resource=https://bisque2.ece.ucsb.edu/data_service/00-TGbt6MLRm7VCn4mWmVaQsc)).

### [**WildlifeMapper: Aerial Image Analysis for Multi-Species Detection and Identification**](https://openaccess.thecvf.com/content/CVPR2024/papers/Kumar_WildlifeMapper_Aerial_Image_Analysis_for_Multi-Species_Detection_and_Identification_CVPR_2024_paper.pdf)
[Satish Kumar*](https://www.linkedin.com/in/satish-kumar-81912540/), [Bowen Zhang](), .. , [Jared A. Stabach](https://jaredstabach.com/), [Lacey Hughey](), .. , [B S Manjunath](https://vision.ece.ucsb.edu/people/bs-manjunath).

Official repository of our [**CVPR 2024**](https://openaccess.thecvf.com/content/CVPR2024/papers/Kumar_WildlifeMapper_Aerial_Image_Analysis_for_Multi-Species_Detection_and_Identification_CVPR_2024_paper.pdf) paper.

<img src="./docs/wildlifemapper_github.jpg" width="800">

This repository includes:
* Source code of WildlifeMapper.
* Pre-trained weights for the bounding box detector.
* Scripts to download Mara-Wildlife dataset (Approvals under review)
* Online tool to visualize Mara-Wildlife dataset ([**BisQue**](https://bisque2.ece.ucsb.edu/client_service/view?resource=https://bisque2.ece.ucsb.edu/data_service/00-TGbt6MLRm7VCn4mWmVaQsc))
* Code for custom data preparation for training/testing


![supported versions](https://img.shields.io/badge/python-(3.8--3.10)-brightgreen/?style=flat&logo=python&color=green)
![Library](https://img.shields.io/badge/Library-Pytorch-blue)
![GitHub license](https://img.shields.io/cocoapods/l/AFNetworking)


The repository follows the structure of paper, making it easy to follow and use/extend the work. If this research is helpful to you, please consider citing our paper (bibtex below)

## Citing
If this research is helpful to you, please consider citing our paper:

```
@inproceedings{kumar2024wildlifemapper,
  title={WildlifeMapper: Aerial Image Analysis for Multi-Species Detection and Identification},
  author={Kumar, Satish and Zhang, Bowen and Gudavalli, Chandrakanth and Levenson, Connor and Hughey, Lacey and Stabach, Jared A and Amoke, Irene and Ojwang, Gordon and Mukeka, Joseph and Mwiu, Stephen and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12594--12604},
  year={2024}
}
```

## Usage

### Requirements
- Linux or macOS with Python >= 3.7
- Pytorch >= 1.7.0
- CUDA >= 10.0
- cudNN (compatible with CUDA)

### Installation
1. Clone the repository
2. Install dependencies
```
pip install -r requirements.txt
```



## Dataset

See [here](https://bisque2.ece.ucsb.edu/client_service/view?resource=https://bisque2.ece.ucsb.edu/data_service/00-TGbt6MLRm7VCn4mWmVaQsc) for an overview of the datastet. The sample dataset can be downloaded [here](https://bisque2.ece.ucsb.edu/client_service/view?resource=https://bisque2.ece.ucsb.edu/data_service/00-TGbt6MLRm7VCn4mWmVaQsc). By downloading the datasets you agree that you have read and accepted the terms of the SA-1B Dataset Research License.

We save masks per image as a json file. It can be loaded as a dictionary in python in the below format.

```python
{
    "image"                 : image_info,
    "annotations"           : [annotation],
}

image_info {
    "image_id"              : int,              # Image id
    "width"                 : int,              # Image width
    "height"                : int,              # Image height
    "file_name"             : str,              # Image filename
}

annotation {
    "id"                    : int,              # Annotation id
    "bbox"                  : [x, y, w, h],     # The box around the mask, in XYWH format
    "predicted_iou"         : float,            # The model's own prediction of the mask's quality
    "stability_score"       : float,            # A measure of the mask's quality
}
```


## License
MethaneMapper is released under the UCSB license. Please see the [LICENSE](./LICENSE) file for more information.

## Contributors

The WildlifeMapper project was made possible with the help of many contributors for all over the world: Satish Kumar, Bowen Zhang, Chandrakanth Gudavalli, Connor Levenson, Lacey Hughey, Jared A. Stabach, Irene Amoke, Gordon Ojwang’, Joseph Mukeka, Stephen Mwiu, Joseph Ogutu, Howard Frederick, B.S. Manjunath
