''' Train diffusion model to predict the mean, with longitude-periodic convolutions. '''

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
from wavediffusion.model import Scaled, PredX0
from wavediffusion.wavedata import npyDataWndHist
from wavediffusion.diffusion import ScheduleLogLinear, ScheduleDDPM, samples, masked_training_loop

from wavediffusion.waveutils import evaluate, sample_and_save

OPTION = 1

### TODO: refine this to reuse mean and std stats from model reload. And multi-GPU case for computing dataset stats
def main(path, train_batch_size=1024, epochs=300, sample_batch_size=64, RESUME=False, weights_file=None,
         ckpt_everyn_epoch=1, sample_everyn_epoch=1, eval_everyn_step=100, gradient_accumulation_steps=4):
    # Setup
    print(torch.cuda.is_available())
    a = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=gradient_accumulation_steps)
    print(a.state)

    train_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/mean_global/'
    years = list(range(1993, 2018))
    # years = list(range(2010, 2023))
    train_file_names = [*( (f'wavemean_{y:04d}{m:02d}', f'forcing_{y:04d}{m:02d}') for y in years for m in range(1, 13) )]
    train_file_list = [(os.path.join(train_file_path, f'{x}.npy'),
                        os.path.join(train_file_path, f'{f}.npy')) for x, f in train_file_names]
    stats_file = os.path.join(train_file_path, f'stats_OPTION{OPTION}.npz')
    stats = np.load(stats_file)
    meanx, stdx = stats['meanx'], stats['stdx']
    meanf, stdf = stats['meanf'], stats['stdf']
    train = npyDataWndHist(
        train_file_list,
        resize_x=(320,320), resize_f=(320,320),
        landmaskname=os.path.join(train_file_path, 'mask.npy'),
        use_icymask=True, compute_stats=False,
        meanx=meanx, stdx=stdx, meanf=meanf, stdf=stdf,
        OPTION=OPTION
    )
    test_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/mean_global/'
    test_file_names = [('wavemean_201904', 'forcing_201904')]
    test_file_list = [(os.path.join(test_file_path, f'{x}.npy'),
                       os.path.join(test_file_path, f'{f}.npy')) for x, f in test_file_names]
    test = npyDataWndHist(
        test_file_list,
        resize_x=(320,320), resize_f=(320,320),
        landmaskname=os.path.join(test_file_path, 'mask.npy'),
        use_icymask=True, compute_stats=False,
        meanx=meanx, stdx=stdx, meanf=meanf, stdf=stdf,
        OPTION=OPTION
    )

    loader = DataLoader(train, batch_size=train_batch_size, shuffle=True)
    loader_test = DataLoader(test, batch_size=sample_batch_size, shuffle=True)  # Used for generating samples during training

    schedule_infer = ScheduleLogLinear(sigma_min=0.01, sigma_max=100, N=80)
    # schedule_train = ScheduleLogLinear(sigma_min=80, sigma_max=200, N=10)
    # schedule_infer = ScheduleDDPM()
    schedule_train = ScheduleDDPM()

    # in_ch: number of predicted quantities
    # out_ch: number of predicted quantities
    # precond_ch: number of conditional fields
    # periodic=True: convolutions wrap circularly along the longitude (width) axis
    # instead of zero-padding, since the last column and first column of the
    # global grid are physically adjacent.
    model = Scaled(myUnet)(in_dim=320, in_ch=3, out_ch=3, ch=256, precond_ch=14,
                           scale=(train.meanx, train.stdx, train.meanf, train.stdf),
                           ch_mult=(1, 2, 2), attn_resolutions=(16,), periodic=True)

    # Train
    log_file = open(path + "loss_log.txt", "a")
    test_log_file = open(path + "test_loss_log.txt", "a")
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
        if a.sync_gradients:
            ema.update()

        # ---- evaluation ----
        if ns.step % eval_everyn_step == 0. and ns.step > 0:
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
            if ns.epoch % sample_everyn_epoch == 0:
                a.wait_for_everyone()
                if a.is_main_process:
                    print('Sampling... at epoch ', ns.epoch)
                    sample_and_save(model, ema, loader_test, schedule_infer, a, path, sample_batch_size,
                                    test=test, filename=f"sample_epoch{ns.epoch}", OPTION=OPTION)
                a.wait_for_everyone()

    log_file.close()
    test_log_file.close()

    if a.distributed_type != "NO" and dist.is_available() and dist.is_initialized():
        a.wait_for_everyone()
    a.end_training()


if __name__=='__main__':

    path = f'/global/homes/j/jiarongw/scratch_folder/review/OPTION{OPTION}_periodic/'
    os.makedirs(path, exist_ok=True)
    main(path, train_batch_size=4, epochs=10, sample_batch_size=2, RESUME=False, weights_file=None,
         ckpt_everyn_epoch=1, sample_everyn_epoch=1, gradient_accumulation_steps=4)
