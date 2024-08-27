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

# import sys; sys.path.append(f"/home/{os.getlogin()}/code/NLSPN_ECCV20/")
# from summary import get as get_summary
# from metric import get as get_metric
import sys; sys.path.append(f"/home/{os.getlogin()}/code/PENet_ICRA2021/")
import criteria

## Parse CLI arguments ##
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset', default='NYUv2', help='Dataset to use')
    return parser.parse_args()


def load_data(args):
    # Load dataset
    if args.dataset == 'NYUv2':
        dataset_train = NYUv2Dataset(args=args, mode='train')
        dataset_val = NYUv2Dataset(args=args, mode='val')
        dataset_test = NYUv2Dataset(args=args, mode='test')
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
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False) if dataset_test else None
    return {"loader_train": loader_train, "loader_val": loader_val, "loader_test": loader_test}


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
    
    # from mmcv.utils import Config
    from mmengine.config import Config
    config_file = f'config/{args.dataset}.py'
    cfg_dict_from_file, cfg_text, _ = Config._file2dict(filename=config_file, use_predefined_variables=True)

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
    data_loader = load_data(args)
    loader_train = data_loader["loader_train"]
    loader_val = data_loader["loader_val"]
    loader_test = data_loader["loader_test"]

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
    model, optimizer, loader_train = accelerator.prepare(
        model, optimizer, loader_train
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

    ## Test only ##
    if args.test_only:
        # do sth
        pass
        # return

    ## Iterate across training loop
    for epoch in range(args.epochs):
        print("-"*30 + f"Epoch {epoch+1}/{args.epochs} Start...")

        ## TRAIN ##
        cur_mode = 'train'
        print("Training")

        running_loss = 0.0; counter = 0
        for (img, mask) in tqdm(loader_train):
            with accelerator.accumulate(model):
                loss = diffusion(mask, img)
                accelerator.log({'train_loss': loss})  # Log loss to wandb
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item() * img.size(0); counter +=img.size(0)
        epoch_loss = running_loss / counter
        print(f"Epoch {epoch+1}/{args.epochs} Training End. Average Training Loss: {epoch_loss:.4f}")
        
        ## SAVE CHECKPOINT ##
        if accelerator.is_main_process and (epoch+1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffusion.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(checkpoint_dir, f'state_dict_epoch_{epoch}_loss_{epoch_loss}.pt'))
            
            # save one image per batch
            pred = diffusion.sample(img).cpu().detach().numpy() # inference
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    tracker.log(
                        {f'{cur_mode}_pred-img-mask': [wandb.Image(pred[0, 0, :, :]), wandb.Image(img[0, :3, :, :]),
                                           wandb.Image(mask[0, 0, :, :])]}
                    )
        

        ## EVALUATION ##
        ## TEST ##
        loader_eval = loader_test
        if accelerator.is_main_process and (epoch+1) % args.eval_every == 0:
            cur_mode = 'val'
            print("Evaluation")

            torch.set_grad_enabled(False)
            diffusion.eval()

            running_metric = 0.0; counter = 0
            for (img, mask) in tqdm(loader_eval):
                pred = diffusion.sample(img)    # inference
                accelerator.log({f'{cur_mode}_pred-img-mask': 
                                 [wandb.Image(pred[0, 0, :, :]), wandb.Image(img[0, :3, :, :]), wandb.Image(mask[0, 0, :, :])]})
                
                ## Metric ##
                depth_criterion = criteria.MaskedMSELoss().to(pred.device)
                metric = depth_criterion(pred, mask.to(pred.device)).cpu().detach().numpy()
                accelerator.log({'val_metric_step': metric})  # Log loss to wandb
                running_metric += metric; counter +=img.size(0)
                # print(f"metric.item(): {metric.item()}")    # metric.item(): 5.129260540008545
                # print(f"metric: {metric}")  # metric: 5.129260540008545
            epoch_metric = running_metric / counter
            accelerator.log({'val_metric_epoch': epoch_metric})  # Log loss to wandb
            print(f"Epoch {epoch+1}/{args.epochs} Evaluation End. Average Evaluation Metric: {epoch_metric:.4f}")

            # metric = get_metric(args)
            # metric = metric(args)
            # summary = get_summary(args)
            # writer_val = summary(args.save_dir, 'val', args, loss.loss_name, metric.metric_name)
            # for (img, mask) in tqdm(loader_val):
            #     pred = diffusion.sample(img).cpu().detach().numpy()             # inference
            #     metric_val = metric.evaluate({"gt":mask}, {"pred":pred}, 'val') # metric
            #     writer_val.add(loss_val, metric_val)
            # writer_val.update(epoch, sample, output)
            # # writer_val.save(epoch, batch, sample, output)

            torch.set_grad_enabled(True)



if __name__ == '__main__':
    main()
