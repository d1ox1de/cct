# Description
This repository implements multi-label classification based on [compact convolutional transformers](https://github.com/SHI-Labs/Compact-Transformers) in Pytorch. There are a substantial number of examples of PyTorch models on the Internet, which are either too primitive to extrapolate to real-world problems or too complex to go through them. This repository is a simple yet full-fledged example of a neural network that can help grasp the gist of building more sophisticated models and training procedures. Here one can find how to make use of **transfer learning**, write a **custom loss function**, **custom datasets** with oversampling and various augmentations, and much more.

## Installation
><details><summary> <b>Docker (recommended)</b> </summary>
>
> ``` shell
> # create a docker container, you can change shm (shared memory) if you have more
> docker run --gpus all -it --name cct --shm-size=10g -v {path_to_cloned_repository}:/cct/ -v {path_to_data_dir}:/cct/datasets/ nvcr.io/nvidia/pytorch:21.05-py3
>
># go to src
> cd /cct/src
>```
>
><details><summary><small>example<small></summary>
>
> ``` shell
>docker run --gpus all -it --name cct --shm-size=10g -v /mnt/d/cct/:/cct/ -v /mnt/d/datasets/cats/:/cct/datasets/ nvcr.io/nvidia/pytorch:21.05-py3
>
>```
>
</details>
</details>


><details><summary> <b>Conda</b> </summary>
>
> 1. Create a conda environment and activate it.
>
>``` shell
>conda create --name cct python=3.8 -y
>conda activate cct
>```
>
> 2. Install PyTorch following [official instructions](https://pytorch.org/get-started/locally). Make sure that your CUDA drives are compatible with the CUDA Toolkit you are to install (see [cuda-toolkit](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)). This repository was tested with `pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0`.
>
> 3. Download [this](https://github.com/d1ox1de/cct.git) repository and install required packages.
>
>``` shell
> git clone https://github.com/d1ox1de/cct.git
> cd cct
> pip install -r requirements.txt
> cd src
>```
> 
</details>



## Dataset Preparation
1. Put images in the `datasets` folder. 
2. Annotate the images and provide `train.csv` and `val.csv` in the format described below.

><details><summary><b>format of train.csv (val.csv)<b></summary>
>
>Image paths in the "filename" field must be **absolute paths**. You can change the number and names of custom labels ("label1", "label2", ..., "labeln"), but the `"no_object"` field must always be present. It denotes whether there is one of the labels (from "label1" to "labeln") in an image. `"no_object"=1` means there are no labels at all (so all other labels are set to 0). `"no_object"=0` means there is at least one label in an image.
>
>| filename 				 | no_object  | label1   | label2 |  ...   | labeln |
>| :---     				 | :----:     | :----:	 | :----: |	:----: |:----:  |
>| /cct/datasets/img1.jpg    | 1 		  | 0        | 0      |	...	   | 0      |
>| /cct/datasets/img2.jpg    | 0          | 1        | 1      | ...    | 0      |
>| /cct/datasets/img3.jpg    | 0          | 1        | 0      | ...    | 1      |
>| /cct/datasets/img4.jpg    | 0          | 0        | 0      | ...    | 1      |
>
></details>



Expected directory structure:

```
datasets/
|  img1.jpg
|  img2.jpg
|  ...
csv/
|  train.csv
|  val.csv
src/
│  config.yaml
|  train.py
│  ...   
│
README.md
```
You can change the directory structure *whatever you like*, but make sure that new paths to `train.csv` and `val.csv` inside `config.yaml` are correct.


## Training
1. Modify `config.yaml` to specify paths to pretrained weights and csv files to train and validate your neural network. 

2. Modify `dataloader/hyp.yaml` to change augmentation, add class weights for a loss function, or to turn on oversampling (undersampling).

3. Run `python train.py`.

After training all weights and training stats are stored in `runs/cct/`.

## Inference
You will get an output in the form of a csv file stored in the same folder.

``` shell
python inference.py --weights {path_to_weights} --input {path_to_folder}
```

<details><summary> <small>example<small> </summary>

``` shell
# providing and additional flag argument --output you can specify a folder in which the result is stored
python inference.py --weights ./runs/cct_31-12-2022_0/weights/best.pt --input /mnt/d/images --output .
```

</details>


<details><summary> <small>format of result.csv<small> </summary>

| Filename 				 | Label  | Confidence |
| :---     				 | :----: | :----:	   | 			
| /mnt/d/images/img1.jpg | label2 | 0.95       |
| /mnt/d/images/img2.jpg | label2 | 0.79       |
| /mnt/d/images/img2.jpg | label4 | 0.72       |
| /mnt/d/images/img3.jpg | label1 | 0.86       |
| /mnt/d/images/img3.jpg | label3 | 0.56       |

</details>