# RL_one_shotMiniimagenet
One-shot learning sample using Miniimagenet data set. In the embedding module, we use reinforcement learning method.

# Requirements
Python 3.7<br>
Pytorch 1.3.1


# Data
For mini-Imagenet experiments, please download mini-Imagenet(see the ./data/download miniimage dataset) and put it in ./data/mini-Imagenet. Run proc_image.py to preprocess generate train/val/test datasets. (This process method is based on <a href="https://github.com/cbfinn/maml">maml</a>).
<br>mini-Imagenet directory should be created manually, and read the relative path settings of the run proc_image.py for the path setting.
 
# Train
python miniimagenet_train_one_shot.py
