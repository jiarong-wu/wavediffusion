''' Train a deterministic model (forcing -> wave mean fields) as a baseline for
    meandiff_periodic_hist10d.py: same data, same 10-day daily wind history, and
    the same longitude-periodic UNet backbone (DetUnet = myUnet with zeroed
    noisy-input channels and a fixed sigma), trained with a plain masked MSE. '''

import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator
from torch_ema import ExponentialMovingAverage as EMA
import torch.distributed as dist

from wavediffusion.model_unet import DetUnet
from wavediffusion.plain_unet import masked_training_loop_plain, evaluate_plain
from wavediffusion.wavedata import npyDataWndHist

OPTION = 1
HIST_DURATION_DAYS = 10
HIST_FREQ_DAYS = 1

def main(path, train_batch_size=1024, epochs=300, sample_batch_size=64, RESUME=False, weights_file=None,
         ckpt_everyn_epoch=1, eval_everyn_step=100, gradient_accumulation_steps=4):
    # Setup
    print(torch.cuda.is_available())
    a = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=gradient_accumulation_steps)
    print(a.state)

    train_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/mean_global/'
    years = list(range(1993, 2018))
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
        OPTION=OPTION, hist_duration_days=HIST_DURATION_DAYS, hist_freq_days=HIST_FREQ_DAYS
    )
    test_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/mean_global/'
    # Test set: all of 2019 (held out from training, 1993-2017), every TEST_STRIDE-th sample (see loader_test)
    test_file_names = [(f'wavemean_2019{m:02d}', f'forcing_2019{m:02d}') for m in range(1, 13)]
    test_file_list = [(os.path.join(test_file_path, f'{x}.npy'),
                       os.path.join(test_file_path, f'{f}.npy')) for x, f in test_file_names]
    test = npyDataWndHist(
        test_file_list,
        resize_x=(320,320), resize_f=(320,320),
        landmaskname=os.path.join(test_file_path, 'mask.npy'),
        use_icymask=True, compute_stats=False,
        meanx=meanx, stdx=stdx, meanf=meanf, stdf=stdf,
        OPTION=OPTION, hist_duration_days=HIST_DURATION_DAYS, hist_freq_days=HIST_FREQ_DAYS
    )

    loader = DataLoader(train, batch_size=train_batch_size, shuffle=True)
    # Every 12th sample of 2019 (~240, one every 1.5 days at varying hours): a whole year at April's cost,
    # since evaluate() runs over the whole loader every eval_everyn_step steps.
    TEST_STRIDE = 12
    loader_test = DataLoader(Subset(test, range(0, len(test), TEST_STRIDE)), batch_size=sample_batch_size, shuffle=True)  # Used for validation loss during training

    # in_ch / out_ch: number of predicted quantities (in_ch only sizes the zeroed noisy-input channels)
    # precond_ch: number of conditional fields (4 base forcing channels + 2 per historical wind sample)
    # periodic=True: convolutions wrap circularly along the longitude (width) axis
    precond_ch = 4 + 2 * (HIST_DURATION_DAYS // HIST_FREQ_DAYS)
    model = DetUnet(in_dim=320, in_ch=3, out_ch=3, ch=256, precond_ch=precond_ch,
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
        start_epoch = ckpt["epoch"]
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
                val_loss = evaluate_plain(model, ema, loader_test, a)
                test_log_file.write(f"{ns.step}, {val_loss.item():.6f}\n")
                test_log_file.flush()
            a.wait_for_everyone()

        # ---- epoch-based triggers ----
        if ns.epoch != last_epoch:
            # last_epoch just finished — save its checkpoint before moving on
            if last_epoch >= 0:
                a.wait_for_everyone()

                # ---- checkpoint ----
                if last_epoch % ckpt_everyn_epoch == 0:
                    if a.is_main_process:
                        print('Saving checkpoint... at epoch ', last_epoch)
                        a.save({"model": a.unwrap_model(model).state_dict(), "ema": ema.state_dict(), "epoch": last_epoch},
                                path + f"ckpt_{last_epoch}.pt")
                    a.wait_for_everyone()

            last_epoch = ns.epoch

    # Save final checkpoint after all epochs complete (the loop only saves at epoch starts,
    # so the last epoch's training would otherwise be lost)
    final_epoch = start_epoch + epochs
    a.wait_for_everyone()
    if a.is_main_process:
        print(f'Saving final checkpoint at epoch {final_epoch}')
        a.save({"model": a.unwrap_model(model).state_dict(), "ema": ema.state_dict(), "epoch": final_epoch},
                path + f"ckpt_{final_epoch}.pt")
    a.wait_for_everyone()

    log_file.close()
    test_log_file.close()

    if a.distributed_type != "NO" and dist.is_available() and dist.is_initialized():
        a.wait_for_everyone()
    a.end_training()


if __name__=='__main__':

    path = f'/global/homes/j/jiarongw/scratch_folder/review/OPTION{OPTION}_periodic_hist10d_det/'
    os.makedirs(path, exist_ok=True)
    # main(path, train_batch_size=4, epochs=10, sample_batch_size=2, RESUME=False, weights_file=None,
    #      ckpt_everyn_epoch=1, gradient_accumulation_steps=4)
    main(path, train_batch_size=4, epochs=6, sample_batch_size=2, RESUME=True,
         weights_file=path+'ckpt_16.pt', ckpt_everyn_epoch=1, gradient_accumulation_steps=4)
