## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## Dataset

See [here]() for an overview of the datastet. The dataset can be downloaded [here](). By downloading the datasets you agree that you have read and accepted the terms of the SA-1B Dataset Research License.

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

The model is licensed under the [MIT](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The WildlifeMapper project was made possible with the help of many contributors for all over the world:


## Citing WildlifeMapper

If you use WildlifeMapper in your research, please use the following BibTeX entry.

```
@inproceedings{kumar2024wildlifemapper,
  title={WildlifeMapper: Aerial Image Analysis for Multi-Species Detection and Identification},
  author={Kumar, Satish and Zhang, Bowen and Gudavalli, Chandrakanth and Levenson, Connor and Hughey, Lacey and Stabach, Jared A and Amoke, Irene and Ojwang, Gordon and Mukeka, Joseph and Mwiu, Stephen and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12594--12604},
  year={2024}
}
```
