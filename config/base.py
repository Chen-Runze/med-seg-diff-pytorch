report_to = 'wandb'     # Where to log to. Currently only supports wandb
output_dir = './io/output'
logging_dir = 'logs'
mixed_precision = 'no'  # Whether to do mixed precision, choices=["no", "fp16", "bf16"]

# dataset = 'NYUv2'
# data_path = './io/NYUv2/'
# image_size = (224, 224)
input_img_channels = 3
mask_channels = 1

dim = 64
dim_mults = (1, 2, 4, 8)
self_condition = False  # Whether to do self condition
timesteps = 1000

scale_lr = False        # Whether to scale lr
learning_rate = 1e-4
adam_beta1 = 0.95       # The beta1 parameter for the Adam optimizer
adam_beta2 = 0.999      # The beta2 parameter for the Adam optimizer
adam_weight_decay = 1e-06   # Weight decay magnitude for the Adam optimizer
adam_epsilon = 1e-08    # Epsilon value for the Adam optimizer
use_lion = False        # use Lion optimizer

epochs = 151
batch_size = 8
gradient_accumulation_steps = 4     # The number of gradient accumulation steps
save_every = 10                     # save_every n epochs (default: 10)
load_model_from = None              # path to pt file to load from
