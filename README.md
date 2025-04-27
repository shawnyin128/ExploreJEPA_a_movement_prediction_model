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

# Result:
normal loss: 2.9636642932891846 \
wall loss: 6.656404495239258

# Reference
https://github.com/alexnwang/DL25SP-Final-Project