import torch
from tqdm import tqdm

def to_device(data, device):
    for key, value in data.items():

        if isinstance(value, dict):
            for k, v in value.items():
                data[key][k] = v.to(device)
        else:
            data[key] = value.to(device)
            
    return data


def save(epoch, model, optimizer, scheduler, trnloss, valloss, path):
    state = {'epoch': epoch,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(),
             'trnloss': trnloss,
             'valloss': valloss}
        
    torch.save(state, path)

def train(model, loader, optimizer, criterion, device):
    
    losses = []
    model.train()
    for data in tqdm(loader):
        data = to_device(data, device)

        optimizer.zero_grad()
        output = model(data['input'])
        loss = criterion(output, data['label'])
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
    return losses


def eval(model, loader, criterion, device):
        
    losses = []
    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        for data in tqdm(loader):
            data = to_device(data, device)

            output = model(data['input'])
            loss = criterion(output, data['label'])
            losses.append(loss.item())

            predicted = torch.argmax(output.data, dim=1)
            total += data['label'].size(0)
            correct += (predicted == data['label']).sum()
        
    return losses, total, correct

import torch.nn as nn
def set_train_config(cfg, model, loader, criterion=nn.CrossEntropyLoss()):

    optimizer = set_optimizer(cfg, model.parameters())
    scheduler = set_scheduler(cfg, optimizer)

    
    train_config = {'device': cfg.device,
                    'epochs': cfg.epochs,
                    'model' : model,
                    'loader' : loader,
                    'optimizer': optimizer,
                    'criterion': criterion,
                    'scheduler': scheduler,
                    'save_interval': cfg.save_interval,
                    'save_path': cfg.save_path}
    

    if cfg.checkpoint is not None:
        checkpoint = torch.load(cfg.checkpoint)
        train_config['model'].load_state_dict(checkpoint['model'])
        train_config['optimizer'].load_state_dict(checkpoint['optimizer'])
        train_config['scheduler'].load_state_dict(checkpoint['scheduler'])
        train_config['epochs'] = checkpoint['epoch']
        print(f"load checkpoint")
    
    print(f'Train Configs: {train_config}')
    return train_config




def set_optimizer(cfg, parameters):

    scaled_lr = cfg.lr * 256/cfg.batch_size if cfg.scale_lr else cfg.lr

    if cfg.opt == "sgd":
        optimizer = torch.optim.SGD(parameters,
                                    lr=scaled_lr,
                                    momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay,
                                    nesterov="nesterov" in cfg.opt)
    elif cfg.opt == "rmsprop":
        optimizer = torch.optim.RMSprop(parameters, 
                                        lr=scaled_lr, 
                                        momentum=cfg.momentum, 
                                        weight_decay=cfg.weight_decay, 
                                        eps=0.0316, alpha=0.9)
    elif cfg.opt == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=scaled_lr, weight_decay=cfg.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {cfg.opt}. Only SGD, RMSprop and AdamW are supported.")

    return optimizer


def set_scheduler(cfg, optimizer):

    if cfg.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                            step_size=cfg.lr_step_size, 
                                                            gamma=cfg.lr_gamma)
    elif cfg.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                       T_max=cfg.epochs - cfg.lr_warmup_epochs, 
                                                                       eta_min=cfg.lr_min)
    elif cfg.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                                   gamma=cfg.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{cfg.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported.")


    if cfg.lr_warmup_epochs > 0:
        if cfg.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                                    start_factor=cfg.lr_warmup_decay, 
                                                                    total_iters=cfg.lr_warmup_epochs)
        elif cfg.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 
                                                                      factor=cfg.lr_warmup_decay, 
                                                                      total_iters=cfg.lr_warmup_epochs)
        else:
            raise RuntimeError(f"Invalid warmup lr method '{cfg.lr_warmup_method}'. Only linear and constant are supported.")
        
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, 
                                                             schedulers=[warmup_lr_scheduler, main_lr_scheduler], 
                                                             milestones=[cfg.lr_warmup_epochs])
    else:
        lr_scheduler = main_lr_scheduler


    return lr_scheduler