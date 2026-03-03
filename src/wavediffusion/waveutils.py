from matplotlib import pyplot as plt

# Plotting sample v.s. truth, and forcing (only wind vectors for now)
# x: sampled wave data, shape (C, H, W)
# x_true: true wave data, shape (C, H, W)
# f: forcing data, shape (C_f, H, W)
def plot_sample(x_true, x, f, VAR='lp'):
    n_sample = x.shape[0]
    
    if x.shape[1] == 4:
        fig, axes = plt.subplots(1+n_sample, 4, figsize=[32, 3*(1+n_sample)], dpi=100)
        titles = ['wave height','wave length','direction','spread','wind u','wind v']
        imgs = [
            axes[0, 0].imshow(x_true[0], vmin=0, vmax=10, cmap='Blues'),
            axes[0, 1].imshow(x_true[1], vmin=0, vmax=300, cmap='Reds'),
            axes[0, 2].imshow(x_true[2], vmin=0, vmax=360, cmap='twilight_r'),
            axes[0, 3].imshow(x_true[3], vmin=0, vmax=90, cmap='Grays'),
            # axes[0, 4].imshow(f[0], vmin=-20, vmax=20, cmap='PiYG'),
            # axes[0, 5].imshow(f[1], vmin=-20, vmax=20, cmap='PiYG'),
        ]
        axes[0, 0].set_ylabel(f"Truth", fontsize=14, rotation=0, labelpad=40)
        cbars = []
        for j in range(4):
            axes[0,j].set_xticks([]); axes[0,j].set_yticks([]) 
            axes[0,j].set_title(titles[j])
            cbar = fig.colorbar(imgs[j], ax=axes[0,j], fraction=0.046, pad=0.04)
            cbars.append(cbar)
        cbars[1].set_ticks([0,100,200,300]) 
        cbars[2].set_ticks([0,90,180,270,360])    
            
        for i in range(0, n_sample):
            imgs_sample =[
                axes[i+1, 0].imshow(x[i][0], vmin=0, vmax=10, cmap='Blues'),
                axes[i+1, 1].imshow(x[i][1], vmin=0, vmax=300, cmap='Reds'),
                axes[i+1, 2].imshow(x[i][2], vmin=0, vmax=360, cmap='twilight_r'),
                axes[i+1, 3].imshow(x[i][3], vmin=0, vmax=90, cmap='Grays'),
                # axes[i+1, 4].imshow(f[0], vmin=-20, vmax=20, cmap='PiYG'),
                # axes[i+1, 5].imshow(f[1], vmin=-20, vmax=20, cmap='PiYG'),
            ]
            axes[i+1, 0].set_ylabel(f"Sample {i}", fontsize=14, rotation=0, labelpad=40)
            cbars = []
            for j in range(4):
                axes[i+1,j].set_xticks([]); axes[i+1,j].set_yticks([]) 
                cbar = fig.colorbar(imgs_sample[j], ax=axes[i+1,j], fraction=0.046, pad=0.04)
                cbars.append(cbar)
            cbars[1].set_ticks([0,100,200,300]) 
            cbars[2].set_ticks([0,90,180,270,360]) 
        
        plt.tight_layout(rect=[0.06, 0, 1, 1])  # leave room on the left for text
        
    elif x.shape[1] == 1: 
        if VAR == 'lp':
            fig, axes = plt.subplots(1+n_sample, 1, figsize=[8, 3*(1+n_sample)], dpi=100)
            titles = ['wave height','wave length','direction','spread','wind u','wind v']
            imgs = [
                axes[0].imshow(x_true[0], vmin=0, vmax=300, cmap='Reds'),
            ]
            axes[0].set_ylabel(f"Truth", fontsize=14, rotation=0, labelpad=40)
                
            for i in range(0, n_sample):
                imgs_sample =[
                    axes[i+1].imshow(x[i][0], vmin=0, vmax=300, cmap='Reds'),
                ]
                axes[i+1].set_ylabel(f"Sample {i}", fontsize=14, rotation=0, labelpad=40)
            
            plt.tight_layout()          
         
    return fig

def plot_forcing(f):    
    fig, axes = plt.subplots(1, 3, figsize=[22, 3], dpi=100)
    titles = ['wind u','wind v', 'ice frac']
    imgs = [
        axes[0].imshow(f[0], vmin=-20, vmax=20, cmap='PiYG'),
        axes[1].imshow(f[1], vmin=-20, vmax=20, cmap='PiYG'),
        axes[2].imshow(1 - f[2], vmin=0, vmax=1, cmap='Purples'),
    ]
    # axes[0, 0].set_ylabel(f"Truth", fontsize=14, rotation=0, labelpad=40)
    cbars = []
    for j in range(3):
        axes[j].set_xticks([]); axes[j].set_yticks([]) 
        axes[j].set_title(titles[j])
        cbar = fig.colorbar(imgs[j], ax=axes[j], fraction=0.046, pad=0.04)
        cbars.append(cbar)
    cbars[0].set_ticks([-20,-10,0,10,20]) 
    cbars[1].set_ticks([-20,-10,0,10,20]) 
    cbars[2].set_ticks([0,0.5,1])
    plt.tight_layout(rect=[0.06, 0, 1, 1])  # leave room on the left for text    

def plot_sample_new(x_true, x, f):
    n_sample = x.shape[0]
    fig, axes = plt.subplots(1+n_sample, 6, figsize=[44, 3*(1+n_sample)], dpi=100)
    titles = ['wave height','wave length','direction','spread','wind u','wind v']
    imgs = [
        axes[0, 0].imshow(x_true[0], vmin=0, vmax=10, cmap='Blues'),
        axes[0, 1].imshow(x_true[1], vmin=0, vmax=300, cmap='Reds'),
        axes[0, 2].imshow(x_true[2], vmin=0, vmax=360, cmap='twilight_r'),
        axes[0, 3].imshow(x_true[3], vmin=0, vmax=90, cmap='Grays'),
        axes[0, 4].imshow(f[0], cmap='PiYG'),
        axes[0, 5].imshow(f[1], cmap='PiYG'),
    ]
    axes[0, 0].set_ylabel(f"Truth", fontsize=14, rotation=0, labelpad=40)
    for j in range(6):
        axes[0,j].set_title(titles[j])
        fig.colorbar(imgs[j], ax=axes[0,j], fraction=0.046, pad=0.04)
        
    for i in range(0, n_sample):
        imgs_sample =[
            axes[i+1, 0].imshow(x[i][0], vmin=0, vmax=10, cmap='Blues'),
            axes[i+1, 1].imshow(x[i][1], vmin=0, vmax=300, cmap='Reds'),
            axes[i+1, 2].imshow(x[i][2], vmin=0, vmax=360, cmap='twilight_r'),
            axes[i+1, 3].imshow(x[i][3], vmin=0, vmax=90, cmap='Grays'),
            axes[i+1, 4].imshow(f[0], cmap='PiYG'),
            axes[i+1, 5].imshow(f[1], cmap='PiYG'),
        ]
        axes[i+1, 0].set_ylabel(f"Sample {i}", fontsize=14, rotation=0, labelpad=40)
        
        for j in range(6):
            fig.colorbar(imgs_sample[j], ax=axes[i+1,j], fraction=0.046, pad=0.04)
    
    plt.tight_layout(rect=[0.06, 0, 1, 1])  # leave room on the left for text
    return fig

def plot_wave(x, label='None'):
    n_sample = x.shape[0]
    fig, axes = plt.subplots(1, 4, figsize=[30, 3], dpi=100)
    titles = ['wave height','wave length','direction','spread']
    imgs = [
        axes[0].imshow(x[0], vmin=0, vmax=10, cmap='Blues'),
        axes[1].imshow(x[1], vmin=0, vmax=300, cmap='Reds'),
        axes[2].imshow(x[2], vmin=0, vmax=360, cmap='twilight_r'),
        axes[3].imshow(x[3], vmin=0, vmax=90, cmap='Grays'),
    ]
    axes[0].set_ylabel(label, fontsize=14, rotation=0, labelpad=40)
    for j in range(4):
        axes[j].set_title(titles[j])
        fig.colorbar(imgs[j], ax=axes[j], fraction=0.046, pad=0.04)
        
    plt.tight_layout(rect=[0.06, 0, 1, 1])  # leave room on the left for text
    return fig


########### Some evaluation and sampling utilities #############
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from types import SimpleNamespace
from typing import Optional, Union, Tuple
from smalldiffusion.diffusion import Schedule
import torchvision.transforms as tf 
from smalldiffusion.diffusion import generate_train_sample, samples

        
# Resume using
# ckpt = torch.load(path, map_location="cpu")
# model.load_state_dict(ckpt["model"])
# ema.load_state_dict(ckpt["ema"])
# start_epoch = ckpt["epoch"]    

@torch.no_grad()
def evaluate(
    model: nn.Module,
    ema: nn.Module,
    loader: DataLoader,
    schedule: Schedule,
    accelerator: Accelerator,
    conditional: bool = True,
    ) -> torch.Tensor:

    model.eval()
    total_loss, count = 0.0, 0   
    with ema.average_parameters():
        for x, f, mask in loader:
            x = x.to(accelerator.device)
            f = f.to(accelerator.device)
            mask = mask.to(accelerator.device)
            x0 = [x, f]
            x0, sigma, eps, cond = generate_train_sample(x0, schedule, conditional)
            mask = mask.to(eps.device)
            eps = eps * mask
            loss = accelerator.unwrap_model(model).get_loss_masked(x0, sigma, eps, mask=mask, cond=cond)
            
            total_loss += loss.item() * x.shape[0]
            count += x.shape[0]
    
    model.train()

    # Gather total_loss and count from all processes
    # device = next(model.parameters()).device
    # total_loss_tensor = torch.tensor(total_loss, device=device)
    # count_tensor = torch.tensor(count, device=device)
    
    # # Sum across all processes
    # total_loss_tensor = accelerator.reduce(total_loss_tensor, reduction="sum")
    # count_tensor = accelerator.reduce(count_tensor, reduction="sum")
    
    # val_loss_tensor = total_loss_tensor / count_tensor
    # return val_loss_tensor
    
    val_loss_tensor = torch.tensor(total_loss / count, device=next(model.parameters()).device)
    return val_loss_tensor

@torch.no_grad()
def sample_and_save(
    model,
    ema,
    loader,
    schedule,
    accelerator,
    path,
    sample_batch_size,
    test, # test dataset for inverting normalization
    filename="sample",):   

    loader_iter = iter(loader)
    with ema.average_parameters():
        x, f, mask = next(loader_iter)
        x = x.to(accelerator.device)
        f = f.to(accelerator.device)
        mask = mask.to(accelerator.device)
        *xt, x0 = samples(
            model, schedule.sample_sigmas(80), gam=1, batchsize=sample_batch_size, 
            accelerator=accelerator, cond=f, mask=mask,
        )

        for i in range(sample_batch_size):
            x0_ = test.invert_x(x0[i]) * tf.Resize((320,720))(mask[i].to(x0))
            x_  = test.invert_x(x[i])  * tf.Resize((320,720))(mask[i].to(x))
            f_  = test.invert_f(f[i])

            fig = plot_sample(
                x_.cpu().numpy()[:, ::-1],
                x0_.unsqueeze(0).cpu().numpy()[:, :, ::-1],
                f_.cpu().numpy()[:, ::-1],
            )
            fig.savefig(path + filename + f"_{i}.png")
            plt.close(fig)
            
from matplotlib import pyplot as plt

# Plotting sample v.s. truth, and forcing (only wind vectors for now)
# x: sampled wave data, shape (C, H, W)
# x_true: true wave data, shape (C, H, W)
# f: forcing data, shape (C_f, H, W)
def plot_sample(x_true, x, f, VAR='lp'):
    n_sample = x.shape[0]
    
    if x.shape[1] == 4:
        fig, axes = plt.subplots(1+n_sample, 4, figsize=[32, 3*(1+n_sample)], dpi=100)
        titles = ['wave height','wave length','direction','spread','wind u','wind v']
        imgs = [
            axes[0, 0].imshow(x_true[0], vmin=0, vmax=10, cmap='Blues'),
            axes[0, 1].imshow(x_true[1], vmin=0, vmax=300, cmap='Reds'),
            axes[0, 2].imshow(x_true[2], vmin=0, vmax=360, cmap='twilight_r'),
            axes[0, 3].imshow(x_true[3], vmin=0, vmax=90, cmap='Grays'),
            # axes[0, 4].imshow(f[0], vmin=-20, vmax=20, cmap='PiYG'),
            # axes[0, 5].imshow(f[1], vmin=-20, vmax=20, cmap='PiYG'),
        ]
        axes[0, 0].set_ylabel(f"Truth", fontsize=14, rotation=0, labelpad=40)
        cbars = []
        for j in range(4):
            axes[0,j].set_xticks([]); axes[0,j].set_yticks([]) 
            axes[0,j].set_title(titles[j])
            cbar = fig.colorbar(imgs[j], ax=axes[0,j], fraction=0.046, pad=0.04)
            cbars.append(cbar)
        cbars[1].set_ticks([0,100,200,300]) 
        cbars[2].set_ticks([0,90,180,270,360])    
            
        for i in range(0, n_sample):
            imgs_sample =[
                axes[i+1, 0].imshow(x[i][0], vmin=0, vmax=10, cmap='Blues'),
                axes[i+1, 1].imshow(x[i][1], vmin=0, vmax=300, cmap='Reds'),
                axes[i+1, 2].imshow(x[i][2], vmin=0, vmax=360, cmap='twilight_r'),
                axes[i+1, 3].imshow(x[i][3], vmin=0, vmax=90, cmap='Grays'),
                # axes[i+1, 4].imshow(f[0], vmin=-20, vmax=20, cmap='PiYG'),
                # axes[i+1, 5].imshow(f[1], vmin=-20, vmax=20, cmap='PiYG'),
            ]
            axes[i+1, 0].set_ylabel(f"Sample {i}", fontsize=14, rotation=0, labelpad=40)
            cbars = []
            for j in range(4):
                axes[i+1,j].set_xticks([]); axes[i+1,j].set_yticks([]) 
                cbar = fig.colorbar(imgs_sample[j], ax=axes[i+1,j], fraction=0.046, pad=0.04)
                cbars.append(cbar)
            cbars[1].set_ticks([0,100,200,300]) 
            cbars[2].set_ticks([0,90,180,270,360]) 
        
        plt.tight_layout(rect=[0.06, 0, 1, 1])  # leave room on the left for text
        
    elif x.shape[1] == 1: 
        if VAR == 'lp':
            fig, axes = plt.subplots(1+n_sample, 1, figsize=[8, 3*(1+n_sample)], dpi=100)
            titles = ['wave height','wave length','direction','spread','wind u','wind v']
            imgs = [
                axes[0].imshow(x_true[0], vmin=0, vmax=300, cmap='Reds'),
            ]
            axes[0].set_ylabel(f"Truth", fontsize=14, rotation=0, labelpad=40)
                
            for i in range(0, n_sample):
                imgs_sample =[
                    axes[i+1].imshow(x[i][0], vmin=0, vmax=300, cmap='Reds'),
                ]
                axes[i+1].set_ylabel(f"Sample {i}", fontsize=14, rotation=0, labelpad=40)
            
            plt.tight_layout()          
         
    return fig

def plot_forcing(f):    
    fig, axes = plt.subplots(1, 3, figsize=[22, 3], dpi=100)
    titles = ['wind u','wind v', 'ice frac']
    imgs = [
        axes[0].imshow(f[0], vmin=-20, vmax=20, cmap='PiYG'),
        axes[1].imshow(f[1], vmin=-20, vmax=20, cmap='PiYG'),
        axes[2].imshow(1 - f[2], vmin=0, vmax=1, cmap='Purples'),
    ]
    # axes[0, 0].set_ylabel(f"Truth", fontsize=14, rotation=0, labelpad=40)
    cbars = []
    for j in range(3):
        axes[j].set_xticks([]); axes[j].set_yticks([]) 
        axes[j].set_title(titles[j])
        cbar = fig.colorbar(imgs[j], ax=axes[j], fraction=0.046, pad=0.04)
        cbars.append(cbar)
    cbars[0].set_ticks([-20,-10,0,10,20]) 
    cbars[1].set_ticks([-20,-10,0,10,20]) 
    cbars[2].set_ticks([0,0.5,1])
    plt.tight_layout(rect=[0.06, 0, 1, 1])  # leave room on the left for text    

def plot_sample_new(x_true, x, f):
    n_sample = x.shape[0]
    fig, axes = plt.subplots(1+n_sample, 6, figsize=[44, 3*(1+n_sample)], dpi=100)
    titles = ['wave height','wave length','direction','spread','wind u','wind v']
    imgs = [
        axes[0, 0].imshow(x_true[0], vmin=0, vmax=10, cmap='Blues'),
        axes[0, 1].imshow(x_true[1], vmin=0, vmax=300, cmap='Reds'),
        axes[0, 2].imshow(x_true[2], vmin=0, vmax=360, cmap='twilight_r'),
        axes[0, 3].imshow(x_true[3], vmin=0, vmax=90, cmap='Grays'),
        axes[0, 4].imshow(f[0], cmap='PiYG'),
        axes[0, 5].imshow(f[1], cmap='PiYG'),
    ]
    axes[0, 0].set_ylabel(f"Truth", fontsize=14, rotation=0, labelpad=40)
    for j in range(6):
        axes[0,j].set_title(titles[j])
        fig.colorbar(imgs[j], ax=axes[0,j], fraction=0.046, pad=0.04)
        
    for i in range(0, n_sample):
        imgs_sample =[
            axes[i+1, 0].imshow(x[i][0], vmin=0, vmax=10, cmap='Blues'),
            axes[i+1, 1].imshow(x[i][1], vmin=0, vmax=300, cmap='Reds'),
            axes[i+1, 2].imshow(x[i][2], vmin=0, vmax=360, cmap='twilight_r'),
            axes[i+1, 3].imshow(x[i][3], vmin=0, vmax=90, cmap='Grays'),
            axes[i+1, 4].imshow(f[0], cmap='PiYG'),
            axes[i+1, 5].imshow(f[1], cmap='PiYG'),
        ]
        axes[i+1, 0].set_ylabel(f"Sample {i}", fontsize=14, rotation=0, labelpad=40)
        
        for j in range(6):
            fig.colorbar(imgs_sample[j], ax=axes[i+1,j], fraction=0.046, pad=0.04)
    
    plt.tight_layout(rect=[0.06, 0, 1, 1])  # leave room on the left for text
    return fig

def plot_wave(x, label='None'):
    n_sample = x.shape[0]
    fig, axes = plt.subplots(1, 4, figsize=[30, 3], dpi=100)
    titles = ['wave height','wave length','direction','spread']
    imgs = [
        axes[0].imshow(x[0], vmin=0, vmax=10, cmap='Blues'),
        axes[1].imshow(x[1], vmin=0, vmax=300, cmap='Reds'),
        axes[2].imshow(x[2], vmin=0, vmax=360, cmap='twilight_r'),
        axes[3].imshow(x[3], vmin=0, vmax=90, cmap='Grays'),
    ]
    axes[0].set_ylabel(label, fontsize=14, rotation=0, labelpad=40)
    for j in range(4):
        axes[j].set_title(titles[j])
        fig.colorbar(imgs[j], ax=axes[j], fraction=0.046, pad=0.04)
        
    plt.tight_layout(rect=[0.06, 0, 1, 1])  # leave room on the left for text
    return fig


########### Some evaluation and sampling utilities #############
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from types import SimpleNamespace
from typing import Optional, Union, Tuple
from smalldiffusion.diffusion import Schedule
import torchvision.transforms as tf 
from smalldiffusion.diffusion import generate_train_sample, samples

        
# Resume using
# ckpt = torch.load(path, map_location="cpu")
# model.load_state_dict(ckpt["model"])
# ema.load_state_dict(ckpt["ema"])
# start_epoch = ckpt["epoch"]    

@torch.no_grad()
def evaluate(
    model: nn.Module,
    ema: nn.Module,
    loader: DataLoader,
    schedule: Schedule,
    accelerator: Accelerator,
    conditional: bool = True,
    ) -> torch.Tensor:

    model.eval()
    total_loss, count = 0.0, 0   
    with ema.average_parameters():
        for x, f, mask in loader:
            x = x.to(accelerator.device)
            f = f.to(accelerator.device)
            mask = mask.to(accelerator.device)
            x0 = [x, f]
            x0, sigma, eps, cond = generate_train_sample(x0, schedule, conditional)
            mask = mask.to(eps.device)
            eps = eps * mask
            loss = accelerator.unwrap_model(model).get_loss_masked(x0, sigma, eps, mask=mask, cond=cond)
            
            total_loss += loss.item() * x.shape[0]
            count += x.shape[0]
    
    model.train()

    # Gather total_loss and count from all processes
    # device = next(model.parameters()).device
    # total_loss_tensor = torch.tensor(total_loss, device=device)
    # count_tensor = torch.tensor(count, device=device)
    
    # # Sum across all processes
    # total_loss_tensor = accelerator.reduce(total_loss_tensor, reduction="sum")
    # count_tensor = accelerator.reduce(count_tensor, reduction="sum")
    
    # val_loss_tensor = total_loss_tensor / count_tensor
    # return val_loss_tensor
    
    val_loss_tensor = torch.tensor(total_loss / count, device=next(model.parameters()).device)
    return val_loss_tensor

@torch.no_grad()
def sample_and_save_lp(
    model,
    ema,
    loader,
    schedule,
    accelerator,
    path,
    sample_batch_size,
    test, # test dataset for inverting normalization
    filename="sample",):   

    loader_iter = iter(loader)
    with ema.average_parameters():
        x, f, mask = next(loader_iter)
        x = x.to(accelerator.device)
        f = f.to(accelerator.device)
        mask = mask.to(accelerator.device)
        
        # Only generating field 1
        *xt, x0 = samples(
            model, schedule.sample_sigmas(80), gam=1, batchsize=sample_batch_size, 
            accelerator=accelerator, cond=f, mask=mask,
        )

        for i in range(sample_batch_size):
            # Ad-hoc fix: use truth for x[0]
            x0_concat = torch.cat([x[i,[0]], x0[i,[0]], x[i,[2]], x[i,[3]]], dim=0)
            x0_ = test.invert_x(x0_concat) * tf.Resize((320,720))(mask[i].to(x0))
            x_  = test.invert_x(x[i])  * tf.Resize((320,720))(mask[i].to(x))
            f_  = test.invert_f(f[i])

            fig = plot_sample(
                x_.cpu().numpy()[[0], ::-1],
                x0_.unsqueeze(0).cpu().numpy()[:, [0], ::-1],
                f_.cpu().numpy()[:, ::-1],
            )
            fig.savefig(path + filename + f"_{i}.png")
            plt.close(fig)

