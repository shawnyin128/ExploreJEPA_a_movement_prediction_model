# Model structure
Encoder: a CNN network based on ConvNext block. The wall and object are separated into two identical ConvNext networks. \
Decoder: a simple two-layer MLP network.

# Model specification
Total Trainable Parameters: 293,344

# How to train
## config.yaml
base_model: if want to train from scratch, set base_model to False. Otherwise, save your base model to base_model_weights.pth. \
fine_tune: same as base_model. Only difference is that fine tune only has 1 stage and it uses lambda_{}\_ft setup. \
lambda_{}_n: set the lambda for specific loss term and stage n.
## command
`python train.py`

# Result
## Loss
normal loss: 3.0403687953948975 \
wall loss: 6.670853614807129 \
wall_other loss: 6.982694625854492 \
expert loss: 10.202990531921387
# Ranking
No.8 among about 30 teams, with very marginal performance difference with No.4 - No.7 models. But my models' parameter is 10x - 20x fewer than all the models in front (except the No.1 model which uses deterministic algorithm to replace lots of layers).

# Reference
https://github.com/alexnwang/DL25SP-Final-Project
