# Asymmetric Representation Learning

This is the PyTorch implementation of Asymmetric Representation Learning, where capture the asymmetric structure of directed graph by modeling the different roles of receiving and sending.

<center> Illustration of the outgoing embedding and the incoming embedding over a four-node toy graph. </center>
<center><img src="images/inOutEmbedding.png" alt="inOutEmbedding" style="zoom:100%;" /></center>

## Requirements

Our project is developed using Python 3.7, PyTorch 1.5.0 with CUDA10.2. We recommend you to use anaconda for dependency configuration.

Firstly, you need to create an anaconda environment called ```AGNN``` by

```shell
conda create -n AGNN python=3.7
conda activate AGNN
```

Secondly, you need to install ```pytorch``` by

```shell
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
```

Thirdly, you need to download ```torch-scatter``` and ```torch-sparse``` manually in

```url
https://data.pyg.org/whl/torch-1.5.0%2Bcu102/torch_scatter-2.0.5-cp37-cp37m-linux_x86_64.whl
https://data.pyg.org/whl/torch-1.5.0%2Bcu102/torch_sparse-0.6.7-cp37-cp37m-linux_x86_64.whl
```
and install them by
```url
pip install ...
```
You can also install these two packages follow their official instruction.


## Train

### Node-level task
```shell
python ./code/train_node.py --gpu-no 0 --dataset citeseer --epochs 1000 --early_stopping 200 --num_layer 2 --hidden 64 --dropout 0.6 --normalize-features True
python ./code/train_node.py --gpu-no 0 --dataset amazon_photo --epochs 500 --early_stopping 0 --num_layer 2 --hidden 128 --dropout 0.9 normalize-features False
```

### Graph-level task
```shell
python ./code/train_graph.py --gpu-no 0 --model AGNN_share
```

## Results
<center><img src="images/results.png" alt="results" style="zoom:100%;" /></center>
Overall accuracy comparison on node classification between our AGNN without regularization and AGNN with regularization and seven existing methods. The best results are highlighted in boldface and the second in Italian font.


## Acknowledgements
The template is borrowed from Pytorch-Geometric benchmark suite. We thank the authors of following works for opening source their excellent codes, Pytorch-Geometric, GNN-benchmark, DiGCN, DAGNN.