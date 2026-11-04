''' OPTION1 version of partdiff_periodic_hist10d_highsigma.py (mean fields hs, tm,
    thetam instead of partitions).
    Fine-tune the hist10d mean-field diffusion model (meandiff_periodic_hist10d.py,
    ckpt BASE_CKPT) with the eps loss weighted toward high noise levels: w(sigma) =
    min(1 + sigma^2 / SIGMA_DATA^2, W_MAX) per sample (SigmaWeighted in
    wavediffusion.model; EDM's loss weight, capped). Everything else as in
    meandiff_periodic_hist10d.py, with longitude-periodic convolutions,
    using 10 days of daily wind history as forcing.

    Why: the plain OPTION3 model's regional means of Hs are 5-10% low in the
    Southern Ocean, because its denoiser is biased at high sigma, where the plain eps loss
    weights an x0 error b by only (b/sigma)^2 (analysis/paper_figure/so_bias/README.md).
    With this weighting the x0 error counts ~b^2 / SIGMA_DATA^2 for sigma up to
    sigma_data * sqrt(W_MAX - 1) ~ 20, and still more than before above that. In
    so_bias/figures/denoise_bias.png the bias leaves zero around sigma ~ 5 and
    grows most between sigma ~ 20 and 100. The noise levels are still drawn from ScheduleDDPM (half
    above sigma ~ 3.4), unlike EDM's log-normal draw centred at low sigma.
    The weighting adds no parameters: checkpoints load as Scaled(myUnet), so the
    analysis scripts evaluate them like any other hist10d checkpoint. The test
    loss in test_loss_log.txt is the weighted one, so not comparable to the base run's.
    A fresh run starts from BASE_CKPT's model and EMA weights; its epochs count
    from 0 in the new folder. '''

import numpy as np
import os
import re
import torch
from torch.utils.data import DataLoader, Subset
from matplotlib import pyplot as plt
from accelerate import Accelerator
from torch_ema import ExponentialMovingAverage as EMA
from torchvision import transforms as tf
import torch.distributed as dist

from wavediffusion.model_unet import myUnet
from wavediffusion.model import Scaled, SigmaWeighted
from wavediffusion.wavedata import npyDataWndHist
from wavediffusion.diffusion import ScheduleLogLinear, ScheduleDDPM, samples, masked_training_loop

from wavediffusion.waveutils import evaluate, sample_and_save
OPTION = 1
HIST_DURATION_DAYS = 10
HIST_FREQ_DAYS = 1
BASE_DIR = '/global/homes/j/jiarongw/scratch_folder/review/OPTION1_periodic_hist10d/'
BASE_CKPT = 20      # as for OPTION3 (the base run went on to 22)
SIGMA_DATA = 1.0    # x is normalized to ~unit std per channel
W_MAX = 400.0       # cap on w(sigma), reached at sigma ~ 20
LR = 5e-5           # half the base run's 1e-4, for fine-tuning

### TODO: refine this to reuse mean and std stats from model reload. And multi-GPU case for computing dataset stats
def main(path, train_batch_size=1024, epochs=300, sample_batch_size=64, RESUME=False, weights_file=None,
         ckpt_everyn_epoch=1, sample_everyn_epoch=1, eval_everyn_step=100, gradient_accumulation_steps=4):
    # Setup
    print(torch.cuda.is_available())
    a = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=gradient_accumulation_steps)
    print(a.state)

    train_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/mean_global/'
    forcing_file_path = train_file_path
    years = list(range(1993, 2018))
    # years = list(range(2010, 2023))
    train_file_names = [*( (f'wavemean_{y:04d}{m:02d}', f'forcing_{y:04d}{m:02d}') for y in years for m in range(1, 13) )]
    train_file_list = [(os.path.join(train_file_path, f'{x}.npy'),
                        os.path.join(forcing_file_path, f'{f}.npy')) for x, f in train_file_names]
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
    test_file_path = train_file_path
    # Test set: all of 2019 (held out from training, 1993-2017), every TEST_STRIDE-th sample (see loader_test)
    test_file_names = [(f'wavemean_2019{m:02d}', f'forcing_2019{m:02d}') for m in range(1, 13)]
    test_file_list = [(os.path.join(test_file_path, f'{x}.npy'),
                       os.path.join(forcing_file_path, f'{f}.npy')) for x, f in test_file_names]
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
    loader_test = DataLoader(Subset(test, range(0, len(test), TEST_STRIDE)), batch_size=sample_batch_size, shuffle=True)  # Used for generating samples during training

    schedule_infer = ScheduleLogLinear(sigma_min=0.01, sigma_max=100, N=80)
    # schedule_train = ScheduleLogLinear(sigma_min=0.01, sigma_max=100, N=80)
    # schedule_infer = ScheduleDDPM()
    schedule_train = ScheduleDDPM()

    # in_ch: number of predicted quantities
    # out_ch: number of predicted quantities
    # precond_ch: number of conditional fields (4 base forcing channels + 2 per historical wind sample)
    # periodic=True: convolutions wrap circularly along the longitude (width) axis
    # instead of zero-padding, since the last column and first column of the
    # global grid are physically adjacent.
    precond_ch = 4 + 2 * (HIST_DURATION_DAYS // HIST_FREQ_DAYS)
    model = SigmaWeighted(Scaled(myUnet), sigma_data=SIGMA_DATA, w_max=W_MAX)(in_dim=320, in_ch=3, out_ch=3, ch=256, precond_ch=precond_ch,
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
    else:
        ckpt = torch.load(os.path.join(BASE_DIR, f'ckpt_{BASE_CKPT}.pt'), map_location="cpu")
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        if a.is_main_process:
            print(f"Fine-tuning from {BASE_DIR}ckpt_{BASE_CKPT}.pt with w(sigma) = min(1 + sigma^2/{SIGMA_DATA}^2, {W_MAX})")

    ema.to(a.device)

    train_iter = masked_training_loop(
        loader, model, schedule_train,
        lr=LR, epochs=epochs, accelerator=a, conditional=True, start_epoch=start_epoch
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

                # ---- sampling ----
                if last_epoch % sample_everyn_epoch == 0:
                    if a.is_main_process:
                        print('Sampling... at epoch ', last_epoch)
                        sample_and_save(model, ema, loader_test, schedule_infer, a, path, sample_batch_size,
                                        test=test, filename=f"sample_epoch{last_epoch}", OPTION=OPTION)
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

    path = f'/global/homes/j/jiarongw/scratch_folder/review/OPTION{OPTION}_periodic_hist10d_highsigma/'
    os.makedirs(path, exist_ok=True)
    # First run (2026-10-29, job 60452912): fresh from the base run's ckpt 20 -> ckpt_1..3
    # (timed out at 12 h, ~3.2 h per epoch).
    # main(path, train_batch_size=4, epochs=6, sample_batch_size=2, RESUME=False, weights_file=None,
    # ckpt_everyn_epoch=1, sample_everyn_epoch=1, gradient_accumulation_steps=4)
    # Resume from the latest checkpoint up to TARGET_EPOCHS (rerun the job until it gets there).
    TARGET_EPOCHS = 6
    done = max(int(re.match(r'ckpt_(\d+)\.pt$', f).group(1)) for f in os.listdir(path) if re.match(r'ckpt_\d+\.pt$', f))
    main(path, train_batch_size=4, epochs=TARGET_EPOCHS - done, sample_batch_size=2, RESUME=True,
         weights_file=path + f'ckpt_{done}.pt', ckpt_everyn_epoch=1, sample_everyn_epoch=1, gradient_accumulation_steps=4)
