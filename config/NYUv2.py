_base_ = './base.py'

dataset='NYUv2'
data_path='./io/NYUv2/'
split_json = '/home/chenrunze/code/NLSPN_ECCV20/data_json/nyu.json'

image_size = (224,304)      # (228, 304)
num_sample = 500            # number of sparse samples
input_img_channels = 4

# UNet
dim_mults = (1, 2, 4, 8)
timesteps=100

epochs=150
batch_size=5
learning_rate=1e-4

test_only=False
save_every=1
eval_every=1
