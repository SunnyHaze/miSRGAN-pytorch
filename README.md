# Unofficial implementation of miSRGAN (2021 Medical Image Analysis)

``This repository provide a pytorch version implementation of paper "3D Registration of pre-surgical prostate MRI and histopathology images via super-resolution volume reconstruction".


## Dataset Prepare
This paper utilize two public prostate datasets and a internal dataset from Stanford Hospital. Therefore we only provide the link to the public datasets, links are shown below:

- [ProstateX](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=23691656#23691656860763166b154d3b8294e6ff0c206fa5)
- [Prostate Diagnosis](https://wiki.cancerimagingarchive.net/display/Public/PROSTATE-DIAGNOSIS#327725498004a7544e04a10a36cf7ed85def9d0)

Following the paper, we only take a subset with all T2w related volumes to train the model.

## Quick Start
- Remanage dataset
  - We regenerates `.pkl` files to save the volumes before start training.
  - You should follow the python script in `./revise_datasets` dir to generate your own datasets. Pay attention to the output `meta_data.json` file, which is important for Dataloader during training.

- Then you can start the multi GPU training:
```
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=2 \
train_main.py \
    --world_size 2 \
    --data_path "/root/Dataset/prostate/meta_data.json" \
    --batch_size 8 \
    --epochs 70 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --warmup_epochs 2 \
    --output_dir ./output_dir_2/ \
    --log_dir ./output_dir_2/ \
2> train_error_2.log 1>train_logs_2.log
```
You need to change the argument `data_path` to your generated `meta_data.json` file.

## Citation
```
 @article{Sood_Shao_Kunder_Teslovich_Wang_Soerensen_Madhuripan_Jawahar_Brooks_Ghanouni_et al._2021, title={3D Registration of pre-surgical prostate MRI and histopathology images via super-resolution volume reconstruction}, volume={69}, ISSN={13618415}, DOI={10.1016/j.media.2021.101957}, journal={Medical Image Analysis}, author={Sood, Rewa R. and Shao, Wei and Kunder, Christian and Teslovich, Nikola C. and Wang, Jeffrey B. and Soerensen, Simon J.C. and Madhuripan, Nikhil and Jawahar, Anugayathri and Brooks, James D. and Ghanouni, Pejman and Fan, Richard E. and Sonn, Geoffrey A. and Rusu, Mirabela}, year={2021}, month={Apr}, pages={101957}, language={en} }

```
