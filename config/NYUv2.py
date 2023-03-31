dataset='NYUv2'
data_path='./io/NYUv2/'        # /data/chenrunze/dataset/nyudepthv2/
split_json = '/home/chenrunze/code/NLSPN_ECCV20/data_json/nyu.json'

image_size = 224

# UNet
dim_mults = (1, 2, 4, 8)
timesteps=1000

epochs=151
batch_size=7
learning_rate=1e-4

save_every=10
