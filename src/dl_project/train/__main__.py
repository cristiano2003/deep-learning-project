from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
import dl_project
import torch
import wandb
import pytorch_lightning as pl
import argparse
import os

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet',
                    help='model name')
parser.add_argument('--max_epochs', '-me', type=int, default=20,
                    help='max epoch')
parser.add_argument('--batch_size', '-bs', type=int, default=64,
                    help='batch size')
parser.add_argument('--lr', '-l', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--num_workers', '-nw', type=int, default=0,
                    help='number of workers')
parser.add_argument('--seed', '-s', type=int, default=42,
                    help='seed')
parser.add_argument('--wandb', '-w', default=False, action='store_true',
                    help='use wandb or not')
parser.add_argument('--wandb_key', '-wk', type=str,
                    help='wandb API key')
args = parser.parse_args()


def train(args, model_name):
    pl.seed_everything(args.seed, workers=True)

    # WANDB (OPTIONAL)
    if args.wandb:
        wandb.login(key=args.wandb_key)  # API KEY
        name = f"{model_name}-{args.max_epochs}-{args.batch_size}-{args.lr}"
        logger = WandbLogger(project="deep-learning-hust",
                             name=name,
                             log_model="all")
    else:
        logger = None

    # DATALOADER
    train_dataset = dl_project.ASLDataset("train")
    train_dataset, val_dataset = random_split(dataset=train_dataset,
                                              lengths=(0.9, 0.1),
                                              generator=torch.Generator().manual_seed(args.seed))
    test_dataset = dl_project.ASLDataset("test")

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                            num_workers=0,
                            shuffle=False)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             num_workers=0,
                             shuffle=False)

    # MODEL
    model = dl_project.ASLModel(model=model_name, lr=args.lr)

    # CALLBACK
    root_path = os.path.join(os.getcwd(), "checkpoints")
    ckpt_path = os.path.join(os.path.join(root_path, f"{args.model}/"))
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    ckpt_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath=ckpt_path,
        filename=f"{model_name}",
        save_top_k=1,
        mode="max"
    )  # save top 3 epochs with the lowest validation loss
    lr_callback = LearningRateMonitor("step")

    # TRAINER
    trainer = pl.Trainer(default_root_dir=root_path,
                         logger=logger,
                         callbacks=[ckpt_callback, lr_callback],
                         gradient_clip_val=0.5,
                         max_epochs=args.max_epochs,
                         enable_progress_bar=True,
                         deterministic=True,
                         log_every_n_steps=1)

    # FIT MODEL
    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    # TEST MODEL
    trainer.test(dataloaders=test_loader)

    wandb.finish(quiet=True)


if __name__ == '__main__':
    if args.model == "all":
        list_model = ["resnet", "cnn", "vit", "mobilenetv1", "mobilenetv2", "swin"]
        for model_name in list_model:
            train(args, model_name)
    else:
        train(args, args.model)
