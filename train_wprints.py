import click
import os
import classifier_lib
import torch
import numpy as np
import dnnlib
from guided_diffusion.image_datasets import load_data_latent
import random

@click.command()
@click.option('--savedir', help='Save directory', metavar='PATH', type=str, required=True, default="/pretrained_models/discriminator")
@click.option('--gendir', help='Fake sample absolute directory', metavar='PATH', type=str, required=True, default="/gen_latents")
@click.option('--datadir', help='Real sample absolute directory', metavar='PATH', type=str, required=True, default="/real_latents")
@click.option('--img_resolution', help='Image resolution', metavar='INT', type=click.IntRange(min=1), default=256)
@click.option('--pretrained_classifier_ckpt', help='Path of ADM classifier', metavar='STR', type=str, default='/pretrained_models/ADM_classifier/32x32_classifier.pt')
@click.option('--batch_size', help='Num samples', metavar='INT', type=click.IntRange(min=1), default=1024)
@click.option('--epoch', help='Num samples', metavar='INT', type=click.IntRange(min=1), default=60)
@click.option('--lr', help='Learning rate', metavar='FLOAT', type=click.FloatRange(min=0), default=3e-4)
@click.option('--device', help='Device', metavar='STR', type=str, default='cuda:0')

def main(**kwargs):
    print("Inside main function")
    opts = dnnlib.EasyDict(kwargs)
    savedir = os.getcwd() + opts.savedir
    print("Saving to:", savedir)
    os.makedirs(savedir, exist_ok=True)

    print("Preparing data loaders")
    gen_train_loader = load_data_latent(
        data_dir=opts.gendir,
        batch_size=int(opts.batch_size / 2),
        image_size=32,
        class_cond=True,
        random_crop=False,
        random_flip=False,
    )
    real_train_loader = load_data_latent(
        data_dir=opts.datadir,
        batch_size=int(opts.batch_size / 2),
        image_size=32,
        class_cond=True,
        random_crop=False,
        random_flip=False,
    )
    print("Data loaders prepared")

    print("Loading classifier and discriminator")
    pretrained_classifier_ckpt = os.getcwd() + opts.pretrained_classifier_ckpt
    classifier = classifier_lib.load_classifier(pretrained_classifier_ckpt, 32, opts.device, eval=False)
    discriminator = classifier_lib.load_discriminator(None, opts.device, True, eval=False)

    vpsde = classifier_lib.vpsde()
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=opts.lr, weight_decay=1e-7)
    loss = torch.nn.BCELoss()

    iterator = iter(gen_train_loader)
    print("Starting training loop")
    for i in range(opts.epoch):
        print(f"Epoch {i+1} started")
        outs = []
        cors = []
        num_data = 0
        for data in real_train_loader:
            optimizer.zero_grad()
            real_inputs, real_condition = data
            # ... rest of your training code ...

        print(f"Epoch {i+1} completed, saving model")
        torch.save(discriminator.state_dict(), savedir + f"/discriminator_{i+1}.pt")
        print(f"Model saved for epoch {i+1}")

    print("Training completed")

if __name__ == "__main__":
    print("Script started, calling main")
    main()
