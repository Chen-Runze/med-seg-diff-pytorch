import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.optim import AdamW
from lion_pytorch import Lion
from med_seg_diff_pytorch import Unet, MedSegDiff
from med_seg_diff_pytorch.dataset import ISICDataset, GenericNpyDataset, NYUv2Dataset
from accelerate import Accelerator
import wandb

## Parse CLI arguments ##
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset', default='NYUv2', help='Dataset to use')
    return parser.parse_args()


def load_data(args):
    # Load dataset
    if args.dataset == 'NYUv2':
        dataset = NYUv2Dataset(args=args, mode='val')
    elif args.dataset == 'ISIC':
        transform_list = [transforms.Resize(args.image_size), transforms.ToTensor(), ]
        transform_train = transforms.Compose(transform_list)
        dataset = ISICDataset(args.data_path, args.csv_file, args.img_folder, transform=transform_train, training=True,
                              flip_p=0.5)
    elif args.dataset == 'generic':
        transform_list = [transforms.ToPILImage(), transforms.Resize(args.image_size), transforms.ToTensor()]
        transform_train = transforms.Compose(transform_list)
        dataset = GenericNpyDataset(args.data_path, transform=transform_train, test_flag=False)
    else:
        raise NotImplementedError(f"Your dataset {args.dataset} hasn't been implemented yet.")

    ## Define PyTorch data generator
    training_generator = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True)

    return dataset, training_generator


def main():
    args = parse_args()

    def ns_to_dict(namespace):
        """
        Convert an argparse.Namespace object to a dictionary.

        Args:
            namespace (argparse.Namespace): the Namespace object to convert

        Returns:
            dict: a dictionary with the same keys and values as the Namespace object,
                with nested dictionaries created for keys containing a period ('.')
        """
        items = vars(namespace)
        result = {}
        for key, value in items.items():
            if '.' in key:
                prefix, suffix = key.split('.', 1)
                if prefix not in result:
                    result[prefix] = argparse.Namespace()
                setattr(result[prefix], suffix, value)
            else:
                result[key] = value
        for key, value in result.items():
            if isinstance(value, argparse.Namespace):
                result[key] = ns_to_dict(value)
        return result
    
    from mmcv.utils import Config
    config_file = f'config/{args.dataset}.py'
    cfg_dict_from_file, cfg_text = Config._file2dict(filename=config_file, use_predefined_variables=True)

    cfg_from_args = Config(ns_to_dict(args), cfg_text='ns_to_dict', filename='dump_to.py')
    cfg_from_args.merge_from_dict(cfg_dict_from_file)
    args = cfg_from_args

    import time; current_time = time.strftime('%Y-%m-%d@%H-%M')
    checkpoint_dir = os.path.join(args.output_dir, current_time, 'checkpoints')
    logging_dir = os.path.join(args.output_dir, current_time, args.logging_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("med-seg-diff", config=vars(args))
        args.dump(file=os.path.join(args.output_dir, current_time, 'config.py'))

    ## DEFINE MODEL ##
    model = Unet(
        dim=args.dim,
        image_size=args.image_size,
        dim_mults=args.dim_mults,
        mask_channels=args.mask_channels,
        input_img_channels=args.input_img_channels,
        self_condition=args.self_condition
    )

    ## LOAD DATA ##
    dataset, data_loader = load_data(args)
    # training_generator = tqdm(data_loader, total=int(len(data_loader)))
    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.batch_size * accelerator.num_processes
        )

    ## Initialize optimizer
    if not args.use_lion:
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = Lion(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay
        )

    ## TRAIN MODEL ##
    model, optimizer, data_loader = accelerator.prepare(
        model, optimizer, data_loader
    )
    diffusion = MedSegDiff(
        model,
        timesteps=args.timesteps
    ).to(accelerator.device)

    if args.load_model_from is not None:
        save_dict = torch.load(args.load_model_from)
        diffusion.model.load_state_dict(save_dict['model_state_dict'])
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        accelerator.print(f'Loaded from {args.load_model_from}')

    print(accelerator.distributed_type)     # DistributedType.MULTI_GPU
    print(data_loader.__class__)            # <class 'accelerate.data_loader.DataLoaderShard'>
    print(data_loader.__dir__())
    # ['dataset', 'num_workers', 'prefetch_factor', 'pin_memory', 'timeout', 'worker_init_fn',
    #  '_DataLoader__multiprocessing_context', '_dataset_kind', 'batch_size', 'drop_last', 'sampler',
    #  'batch_sampler', 'generator', 'collate_fn', 'persistent_workers', '_DataLoader__initialized',
    #  '_IterableDataset_len_called', '_iterator', 'device', 'rng_types', 'synchronized_generator',
    #  'skip_batches', 'gradient_state', '__module__', '__doc__', '__init__', '__iter__', 'total_batch_size',
    #  'total_dataset_length', '__parameters__', '__annotations__', '_get_iterator', 'multiprocessing_context',
    #  '__setattr__', '_auto_collation', '_index_sampler', '__len__', 'check_worker_number_rationality',
    #  '__orig_bases__', '__dict__', '__weakref__', '__slots__', '_is_protocol', '__class_getitem__',
    #  '__init_subclass__', '__repr__', '__hash__', '__str__', '__getattribute__', '__delattr__', '__lt__',
    #  '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__new__', '__reduce_ex__', '__reduce__',
    #  '__subclasshook__', '__format__', '__sizeof__', '__dir__', '__class__']
    print(data_loader.sampler.__class__)    # <class 'torch.utils.data.sampler.SequentialSampler'>
    print(data_loader.batch_sampler.__class__)  # <class 'accelerate.data_loader.BatchSamplerShard'>

    ## Iterate across training loop
    for epoch in range(args.epochs):
        running_loss = 0.0; counter = 0
        print(f"Epoch {epoch+1}/{args.epochs} Start...")
        for (img, mask) in tqdm(data_loader):
            with accelerator.accumulate(model):
                loss = diffusion(mask, img)
                accelerator.log({'loss': loss})  # Log loss to wandb
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item() * img.size(0); counter +=img.size(0)
            print(f"img.size()={img.size()}")
        epoch_loss = running_loss / counter
        print(f"counter={counter}, len(dataset)={len(dataset)}")
        print(f"Epoch {epoch+1}/{args.epochs} End. Average Training Loss: {epoch_loss:.4f}")
        
        ## SAVE CHECKPOINT ##
        if accelerator.is_main_process and epoch % args.save_every == 0:
        # if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffusion.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(checkpoint_dir, f'state_dict_epoch_{epoch}_loss_{epoch_loss}.pt'))

            ## INFERENCE ##
            pred = diffusion.sample(img).cpu().detach().numpy()

            # if accelerator.is_main_process: proc_id = 'main_process'; print('main_process, pass')
            # else: proc_id = 'sub_process'; print('sub_process, sleep(5)'); import time; time.sleep(5)
            # print(f'---- {proc_id} begin ----')
            # AttributeError: 'Accelerator' object has no attribute 'trackers'
            # print(accelerator.__class__)        # <class 'accelerate.accelerator.Accelerator'>
            for tracker in accelerator.trackers:
                # print(f'---- {proc_id} end ----')
                if tracker.name == "wandb":
                    # save just one image per batch
                    tracker.log(
                        {'pred-img-mask': [wandb.Image(pred[0, 0, :, :]), wandb.Image(img[0, :, :, :]),
                                           wandb.Image(mask[0, 0, :, :])]}
                    )


if __name__ == '__main__':
    main()
