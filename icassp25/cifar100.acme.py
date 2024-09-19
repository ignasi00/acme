#! /usr/bin/python3 -u

import sys
import numpy as np
import importlib

import argparse
import torch

from wide_res_net import WideResNet
from PyramidNet import PyramidNet

from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from mi_data_cifar100 import MiCifar

from torch.optim import SGD
from torch.nn import CrossEntropyLoss

import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None,
        help="Directory with the CIFAR100 database [default: download from repository]", metavar="STR")
    parser.add_argument("--batch-size", default=200, type=int,
        help="Batch size used in the training and validation loop [default: %(default)s].", metavar="INT")
    parser.add_argument("--model", default='WRN', choices=['WRN', 'Pyramid'],
        help="Model to use (WRN or Pyramid) [default: %(default)s].", metavar="STR")
    parser.add_argument("--depth", default=16, type=int,
        help="Number of layers of WRN or Pyramid [default: 40 for WRN, 110 for Pyramid].", metavar="INT")
    parser.add_argument("--width-alpha", default=-1, type=int,
        help="Width of WRN or alpha in Pyramid [default: 8 for WRN, 40 for Pyramid].", metavar="INT")
    parser.add_argument("--bottleneck", action='store_true',
        help="Use bottleneck in Pyramid.")
    parser.add_argument("--dropout", default=0.0, type=float,
        help="Dropout rate [default: %(default)s].", metavar="FLOAT")
    parser.add_argument("--epochs", default=200, type=int,
        help="Total number of epochs [default: %(default)s].", metavar="INT")
    parser.add_argument("--lr", default=0.1, type=float,
        help="Base learning rate at the start of the training [default: %(default)s].", metavar="FLOAT")
    parser.add_argument("--label-smoothing", default=0.1, type=float,
        help="Use 0.0 for no label smoothing [default: %(default)s].", metavar="FLOAT")
    parser.add_argument("--momentum", default=0.9, type=float,
        help="SGD Momentum [default: %(default)s].", metavar="FLOAT")
    parser.add_argument("--threads", default=2, type=int,
        help="Number of CPU threads for dataloaders [default: %(default)s].", metavar="INT")
    parser.add_argument("--rho", default=1.0, type=float,
        help="Rho parameter for SAM [default: %(default)s].", metavar="FLOAT")
    parser.add_argument("--eta", default=0.0, type=float,
        help="Eta parameter for SAM [default: %(default)s].", metavar="FLOAT")
    parser.add_argument("--weight-decay", default=1e-4, type=float,
        help="L2 weight decay [default: %(default).2e].", metavar="FLOAT")
    parser.add_argument("--acme-script", default='acme', type=str,
        help="Name of acme script [default: %(default)s].", metavar="SCRIPT")
    parser.add_argument("--acme", action='store_true',
        help="Use Acme algorithm for learning rate adaption.")
    parser.add_argument("--asam", action='store_true',
        help="Use ASAM optimizer.")
    parser.add_argument("--adam", action='store_true',
        help="Use ADAM optimizer.")
    parser.add_argument("--nesterov", action='store_true',
        help="Use Nesterov accelerated gradient.")
    parser.add_argument("--wandb", default='', type=str,
        help="Name of the wandb run [default: do not use wandb].",
                        metavar="LOGFILE")
    args = parser.parse_args()

    Acme = importlib.import_module(args.acme_script).acme.Acme

    if args.wandb:
        wandb.login()
        wandb.init(project=f"{args.model}", name=args.wandb)

    if args.depth < 0:
        args.depth = 40 if args.model == 'WRN' else 110

    if args.width_alpha < 0:
        args.width_alpha = 8 if args.model == 'WRN' else 40

    initialize(args, seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == 'WRN':
        model = WideResNet(args.depth, args.width_alpha, args.dropout, in_channels=3, labels=100).to(device)
    else:
        model = PyramidNet('cifar100', depth=args.depth, alpha=args.width_alpha, num_classes=100, bottleneck=args.bottleneck).to(device)

    optimizer = Acme(model, lr=args.lr, betas=(args.momentum, 0.999), epsilon=1e-8, weight_decay=args.weight_decay,
        acme=args.acme, adam=args.adam, asam=args.asam, rho=args.rho, eta=args.eta, nesterov=args.nesterov, bias_correction=args.adam)

    criterion = CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda()

    dataset = MiCifar(args.batch_size, args.threads, root=args.data_dir)

    if args.acme:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs, gamma=1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 100)

    log = Log(log_each=10)

    label_smoothing = args.label_smoothing

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            def closure():
                global correct

                with torch.enable_grad():
                    # compute output
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                correct = (torch.argmax(outputs, 1) == targets).sum().div(len(targets))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
        
                return loss

            loss = optimizer.step(closure)

            log(model, torch.tensor([loss]).cpu(), correct.cpu(), optimizer.param_groups[0]['lr'])
        scheduler.step()

        lossEntr = log.epoch_state["loss"]
        correctEntr = log.epoch_state["accuracy"]
        stepsEntr = log.epoch_state["steps"]

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                correct = (torch.argmax(outputs, 1) == targets).sum().div(len(targets))
                log(model, torch.tensor([loss]).cpu(), correct.cpu(), optimizer.param_groups[0]['lr'])

        if args.wandb:
            lossEval = log.epoch_state["loss"]
            correctEval = log.epoch_state["accuracy"]
            stepsEval = log.epoch_state["steps"]

            error = 1 - correctEval / stepsEval

            totCod = 0
            for cod in range(7):
                totCod += optimizer.param_groups[0][cod]
            wandb.log({'lr': optimizer.param_groups[0]['lr'],
                       'lossEntr': lossEntr / stepsEntr,
                       'errEntr': 1 - correctEntr / stepsEntr,
                       'lossEval': lossEval / stepsEval,
                       'errEval': 1 - correctEval / stepsEval,}
                      | ({str(i): optimizer.param_groups[0][i] / totCod * 100 for i in range(7)} if totCod > 0 else {})
                     )
            for cod in range(7):
                optimizer.param_groups[0][cod] = 0

    log.flush()
