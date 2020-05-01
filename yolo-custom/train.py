from models import Darknet
import utils.parse_config as parse_config
import utils.utils as utils
import utils.datasets as datasets

import torch
import torch.utils.data
import torch.autograd
import torch.optim

import time
import datetime
import os
import argparse

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accumulations before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Load data configuration
    data_config = parse_config.parse_data_config(opt.data_config)
    train_path, val_path = data_config['train'], data_config['valid']
    class_names = parse_config.load_classes(data_config['names'])

    # Initialize model
    model = Darknet(opt.model_def).to(device)
    model.apply(utils.weights_init_normal)
    # load weights from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Initialize data loader
    dataset = datasets.ListDataset(train_path)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    average_time_per_epoch = 0
    for epoch in range(opt.epochs):
        model.train()
        starttime = time.time()
        # utils.print_progress_bar(0, len(data_loader), prefix = f'Epoch {epoch}/{opt.epochs}:', suffix = 'Complete', length = 50)
        for batch_i, (_, imgs, targets) in enumerate(data_loader):
            batches_done = len(data_loader) * epoch + batch_i
            
            imgs = torch.autograd.Variable(imgs.to(device))
            targets = torch.autograd.Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                optimizer.step()
                optimizer.zero_grad()

            # utils.print_progress_bar(batch_i + 1, len(data_loader), prefix = f'Epoch {epoch}/{opt.epochs}:', suffix = 'Complete', length = 50)

            model.seen += imgs.size(0)

        epochs_done = epoch + 1
        epochs_left = opt.epochs - epochs_done
        elapsed_time = time.time() - starttime
        average_time_per_epoch *= epoch / epochs_done
        average_time_per_epoch += elapsed_time / epochs_done
        time_left_estimate = datetime.timedelta(seconds=average_time_per_epoch * epochs_left)

        print(f"Epoch {epoch}/{opt.epochs}; total loss {loss.item()}; ETA {time_left_estimate}")

        if epoch % opt.checkpoint_interval == opt.checkpoint_interval - 1:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
