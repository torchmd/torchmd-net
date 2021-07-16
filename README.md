# torchmd-net2

## How to install

```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

conda install pytorch-geometric -c rusty1s -c conda-forge  

OR

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-geometric

pip install git+https://github.com/PyTorchLightning/pytorch-lightning.git@a4abb6248231c357273b9af2bc6c7fa057d79512

pip install .
```

OR

```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

conda install pytorch-geometric -c rusty1s -c conda-forge

pip install git+https://github.com/PyTorchLightning/pytorch-lightning.git@master

pip install .
```

Use the github version of pytorch-lightning to use the beta cli feature (https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html).

## play with the cli

```
cd torchmd-net2
python script/train_2.py --help
python script/train_2.py --conf examples/default_config.yaml

```
