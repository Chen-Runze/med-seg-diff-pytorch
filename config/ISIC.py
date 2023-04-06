_base_ = './base.py'

dataset='ISIC'
data_path='./io/ISIC'
img_folder = 'ISBI2016_ISIC_Part3B_Training_Data',
csv_file = 'ISBI2016_ISIC_Part3B_Training_GroundTruth.csv'
# dataset = dict(
#     data_path='./io/ISIC',
#     img_folder = 'ISBI2016_ISIC_Part3B_Training_Data',
#     csv_file = 'ISBI2016_ISIC_Part3B_Training_GroundTruth.csv'
# )

image_size = (128, 128)

# UNet
dim_mults = (1, 2, 3, 4)
timesteps=1000

epochs=151
batch_size=24
learning_rate=1e-4

save_every=10
