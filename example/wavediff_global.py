import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from accelerate import Accelerator
from torch_ema import ExponentialMovingAverage as EMA
from torchvision import transforms as tf
import torch.distributed as dist

from wavediffusion.model_unet import myUnet
from wavediffusion.model import Scaled
from wavediffusion.wavedata import npyDataResized
from wavediffusion.diffusion import ScheduleLogLinear, ScheduleDDPM, samples, masked_training_loop

from wavediffusion.waveutils import evaluate, sample_and_save

### TODO: refine this to reuse mean and std stats from model reload. And multi-GPU case for computing dataset stats

def main(path, train_batch_size=1024, epochs=300, sample_batch_size=64, RESUME=False, weights_file=None,
         ckpt_everyn_epoch=2, sample_everyn_epoch=1, eval_everyn_step=100):
    # Setup
    print(torch.cuda.is_available())
    a = Accelerator(mixed_precision="fp16") 
    print(a.state)
    
    train_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/stokes_global/'
    train_file_names = [
        *( (f'wave_2010{i:02d}', f'forcing_2010{i:02d}') for i in range(1, 13) ),
        *( (f'wave_2011{i:02d}', f'forcing_2011{i:02d}') for i in range(1, 13) ),
        *( (f'wave_2012{i:02d}', f'forcing_2012{i:02d}') for i in range(1, 13) ),
        *( (f'wave_2013{i:02d}', f'forcing_2013{i:02d}') for i in range(1, 13) ),
        *( (f'wave_2014{i:02d}', f'forcing_2014{i:02d}') for i in range(1, 13) ),
        *( (f'wave_2015{i:02d}', f'forcing_2015{i:02d}') for i in range(1, 13) ),
        *( (f'wave_2016{i:02d}', f'forcing_2016{i:02d}') for i in range(1, 13) ),
        *( (f'wave_2017{i:02d}', f'forcing_2017{i:02d}') for i in range(1, 13) ),
        *( (f'wave_2018{i:02d}', f'forcing_2018{i:02d}') for i in range(1, 13) ),
        *( (f'wave_2019{i:02d}', f'forcing_2019{i:02d}') for i in range(1, 13) ),
        *( (f'wave_2020{i:02d}', f'forcing_2020{i:02d}') for i in range(1, 13) ),
        *( (f'wave_2021{i:02d}', f'forcing_2021{i:02d}') for i in range(1, 13) ),
        *( (f'wave_2022{i:02d}', f'forcing_2022{i:02d}') for i in range(1, 13) ),
    ]
    train_file_list = [(os.path.join(train_file_path, f'{x}.npy'), 
                        os.path.join(train_file_path, f'{f}.npy')) for x, f in train_file_names]   
    stats_file = os.path.join(train_file_path, 'stats.npz')
    stats = np.load(stats_file)
    # Test stokes    
    meanx, stdx = stats['meanx'][[0,4,5]], stats['stdx'][[0,4,5]]
    meanf, stdf = stats['meanf'], stats['stdf']
    # meanx, stdx = stats['meanx'], stats['stdx']
    # meanf, stdf = stats['meanf'], stats['stdf']
    
    train = npyDataResized(
        train_file_list,
        resize_x=(320,320), resize_f=(320,320), 
        landmaskname=os.path.join(train_file_path, 'mask.npy'),
        use_icymask=True, compute_stats=False,
        meanx=meanx, stdx=stdx, meanf=meanf, stdf=stdf
    )
    
    test_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/stokes_global/'
    test_file_names = [('wave_200104', 'forcing_200104')]
    test_file_list = [(os.path.join(test_file_path, f'{x}.npy'), 
                       os.path.join(test_file_path, f'{f}.npy')) for x, f in test_file_names]
    test = npyDataResized(
        test_file_list,
        resize_x=(320,320), resize_f=(320,320), 
        landmaskname=os.path.join(test_file_path, 'mask.npy'),
        use_icymask=True, compute_stats=False,
        meanx=meanx, stdx=stdx, meanf=meanf, stdf=stdf
    )

    loader = DataLoader(train, batch_size=train_batch_size, shuffle=True)
    loader_test = DataLoader(test, batch_size=sample_batch_size, shuffle=True)  # Used for generating samples during training  

    # schedule_infer = ScheduleLogLinear(sigma_min=0.01, sigma_max=60, N=80)
    # schedule_train = ScheduleLogLinear(sigma_min=0.01, sigma_max=100, N=200)
    schedule_infer = ScheduleDDPM()
    schedule_train = ScheduleDDPM()
    
    # in_ch: number of predicted quantities
    # out_ch: number of predicted quantities
    # precond_ch: number of conditional fields
    # model = Scaled(myUnet)(in_dim=320, in_ch=4, out_ch=4, ch=128, precond_ch=3, 
    #                        scale=(train.meanx, train.stdx, train.meanf, train.stdf),
    #                        ch_mult=(1, 2, 2), attn_resolutions=(16,))    
    model = Scaled(myUnet)(in_dim=320, in_ch=3, out_ch=3, ch=192, precond_ch=3, 
                           scale=(train.meanx, train.stdx, train.meanf, train.stdf),
                           ch_mult=(1, 2, 2), attn_resolutions=(16,))    

    # Train
    log_file = open(path + "loss_log.txt", "w")
    test_log_file = open(path + "test_loss_log.txt", "w")
    ema = EMA(model.parameters(), decay=0.999)
    start_epoch = 0
    
    if RESUME and weights_file is not None:
        ckpt = torch.load(weights_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt["epoch"]   # I can pass those to the training loop to count for epochs better
        # start_step = ckpt.get("step", 0)   # Same with steps
        # if a.is_main_process:
        #     print(f"Resuming from epoch {start_epoch}, step {start_step}")
              
    ema.to(a.device)
    
    train_iter = masked_training_loop(
        loader, model, schedule_train,
        lr=1e-4, epochs=epochs, accelerator=a, conditional=True, start_epoch=start_epoch
    )

    last_epoch = -1
    
    for ns in train_iter:
        # ---- logging (only main process) ----
        if a.is_main_process:
            log_file.write(f"{ns.step}, {ns.loss.item():.6f}\n")
            log_file.flush()
            ns.pbar.set_description(f"Loss={ns.loss.item():.5f}")        
            
        ema.update()

        # ---- evaluation ----
        if ns.step % eval_everyn_step == 0:
            a.wait_for_everyone()
            if a.is_main_process:     
                print('Evaluating... at step ', ns.step)
                val_loss = evaluate(model, ema, loader_test, schedule_train, a) # Compute on all GPUs but gather
                test_log_file.write(f"{ns.step}, {val_loss.item():.6f}\n")
                test_log_file.flush()
            a.wait_for_everyone()

        # ---- epoch-based triggers ----
        if ns.epoch != last_epoch:
            last_epoch = ns.epoch
            a.wait_for_everyone()
            
            # ---- checkpoint ----
            if ns.epoch % ckpt_everyn_epoch == 0:
                a.wait_for_everyone()
                if a.is_main_process:
                    print('Saving checkpoint... at epoch ', ns.epoch)
                    a.save({"model": a.unwrap_model(model).state_dict(), "ema": ema.state_dict(), "epoch": ns.epoch},
                            path + f"ckpt_{ns.epoch}.pt")
                a.wait_for_everyone()

            # ---- sampling ----
            # if ns.epoch % sample_everyn_epoch == 0:
            #     a.wait_for_everyone()
            #     if a.is_main_process:
            #         print('Sampling... at epoch ', ns.epoch)
            #         sample_and_save(model, ema, loader_test, schedule_infer, a, path, sample_batch_size, 
            #                         test=test, filename=f"sample_epoch{ns.epoch}")
            #     a.wait_for_everyone()

    log_file.close()
    test_log_file.close()

    if a.distributed_type != "NO" and dist.is_available() and dist.is_initialized():
        a.wait_for_everyone()
    a.end_training()
            
        
if __name__=='__main__':
    path = '/global/homes/j/jiarongw/scratch_folder/log1p/stokes1/'
    main(path, train_batch_size=4, epochs=13, sample_batch_size=2, RESUME=False)    
    # main(path, train_batch_size=8, epochs=13, sample_batch_size=2, RESUME=True, weights_file=path+'ckpt_12.pt')