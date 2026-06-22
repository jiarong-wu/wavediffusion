import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from accelerate import Accelerator
from torch_ema import ExponentialMovingAverage as EMA
from torchvision import transforms as tf
import torch.distributed as dist

from wavediffusion.plain_unet import plainUnet, masked_training_loop_plain, evaluate_plain
from wavediffusion.wavedata import npyDataWndHist

def main(path, train_batch_size=1024, epochs=300, sample_batch_size=64, RESUME=False, weights_file=None,
         ckpt_everyn_epoch=2, eval_everyn_step=100, gradient_accumulation_steps=4):
    # Setup
    print(torch.cuda.is_available())
    a = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=gradient_accumulation_steps) 
    print(a.state)
    
    train_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/train_global/'
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
    meanx, stdx = stats['meanx'], stats['stdx']
    meanf, stdf = stats['meanf'], stats['stdf']
    train = npyDataWndHist(
        train_file_list,
        resize_x=(320,320), resize_f=(320,320), 
        landmaskname=os.path.join(train_file_path, 'mask.npy'),
        use_icymask=True, compute_stats=False,
        meanx=meanx, stdx=stdx, meanf=meanf, stdf=stdf
    )
    test_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/test_global/'
    test_file_names = [('wave_200804', 'forcing_200804')]
    test_file_list = [(os.path.join(test_file_path, f'{x}.npy'), 
                       os.path.join(test_file_path, f'{f}.npy')) for x, f in test_file_names]
    test = npyDataWndHist(
        test_file_list,
        resize_x=(320,320), resize_f=(320,320), 
        landmaskname=os.path.join(test_file_path, 'mask.npy'),
        use_icymask=True, compute_stats=False,
        meanx=meanx, stdx=stdx, meanf=meanf, stdf=stdf
    )

    loader = DataLoader(train, batch_size=train_batch_size, shuffle=True)
    loader_test = DataLoader(test, batch_size=sample_batch_size, shuffle=True)  # Used for generating samples during training  

    model = plainUnet(in_dim=320, in_ch=13, out_ch=4, ch=256, 
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
        if a.is_main_process:
            print(f"Resuming from epoch {start_epoch}")
    ema.to(a.device)

    train_iter = masked_training_loop_plain(loader, model, lr=1e-4, epochs=epochs, 
                                            accelerator=a, start_epoch=start_epoch)

    last_epoch = -1

    for ns in train_iter:
        # ---- logging (only main process) ----
        if a.is_main_process:
            log_file.write(f"{ns.step}, {ns.loss.item():.6f}\n")
            log_file.flush()
            ns.pbar.set_description(f"Loss={ns.loss.item():.5f}")        
        if a.sync_gradients:    
            ema.update()

        # ---- evaluation ----
        if ns.step % eval_everyn_step == 0. and ns.step > 0:
            a.wait_for_everyone()
            if a.is_main_process:     
                print('Evaluating... at step ', ns.step)
                val_loss = evaluate_plain(model, ema, loader_test, a) # Compute on all GPUs but gather
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
    path='/global/homes/j/jiarongw/scratch_folder/log1p/plain_hist/'
    main(path, train_batch_size=4, epochs=4, sample_batch_size=2, RESUME=False, ckpt_everyn_epoch=2, 
         eval_everyn_step=100, gradient_accumulation_steps=4)    
    # main(path, train_batch_size=8, epochs=13, sample_batch_size=2, RESUME=True, weights_file=path+'ckpt_12.pt')