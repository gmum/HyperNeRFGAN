## HyperNeRFGAN: Hypernetwork approach to 3D NeRF GAN

![CARLA](assets/carla.gif) ![ShapeNet](assets/shapenet.gif)

This repo contains HyperNerfGan implementation built on top of the [INR-GAN](https://github.com/universome/inr-gan) repo.
Compared to a traditional generator, ours is [INR](https://vsitzmann.github.io/siren/)-based, i.e. it produces parameters for a fully-connected neural network which implicitly represents a 3D object.

<div style="text-align:center">
<img src="assets/NerfGAN.png" alt="NerfGAN illustration" width="500"/>
</div>

### Installation
To install, run the following command:
```
conda env create --file environment.yaml --prefix ./env
conda activate ./env
```

### Training
To train the model, navigate to the project directory and run:
```
python src/infra/launch_local.py hydra.run.dir=. +experiment_name=my_experiment_name +dataset.name=dataset_name num_gpus=1
```
where `dataset_name` is the name of the dataset without `.zip` extension inside `data/` directory (you can easily override the paths in `configs/main.yml`).
So make sure that `data/dataset_name.zip` exists and should be a plain directory of images.
See [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) repo for additional data format details.
This training command will create an experiment inside `experiments/` directory and will copy the project files into it.
This is needed to isolate the code which produces the model.

### Data format
We use the same data format as the original [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) repo: it is a zip of images.
It is assumed that all data is located in a single directory, specified in `configs/main.yml`.

We also provide downloadable links to some datasets:
- CARLA: The original source is [https://s3.eu-central-1.amazonaws.com/avg-projects/graf/data/carla.zip ](https://s3.eu-central-1.amazonaws.com/avg-projects/graf/data/carla.zip).

Download the datasets and put them into `data/` directory.

### License
This repo is built on top of [INR-GAN](https://github.com/universome/inr-gan) repo, so I assume it is restricted by the [NVidia license](https://nvlabs.github.io/stylegan2-ada-pytorch/license.html).

### Bibtex
```
...
```
