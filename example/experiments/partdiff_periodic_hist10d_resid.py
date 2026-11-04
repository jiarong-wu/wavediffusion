''' Train a diffusion model on the residual of the deterministic partition model:
    x = x_det + rstd * r, with x_det from the frozen hist10d DetUnet
    (partdiff_periodic_hist10d_det.py) and the diffusion model generating r,
    conditioned on the forcing and x_det (ResidualDiffusion in wavediffusion.model).
    Same data, 10-day daily wind history and longitude-periodic UNet as
    partdiff_periodic_hist10d.py.

    Why: the plain diffusion model's regional means of Hs are 5-10% low in the
    Southern Ocean, from a bias of its denoiser at high noise levels that the
    eps-prediction loss barely penalizes (analysis/paper_figure/so_bias/README.md).
    The deterministic model is unbiased there, so here the large-scale level comes
    from it and the diffusion model only adds the variability around it.

    rstd (per-channel std of x - x_det over N_RSTD_SAMPLES training samples,
    with direction residuals wrapped to [-180, 180) deg) scales r to about unit
    variance, which the noise schedule assumes. It is computed once on the main
    process, saved to path/resid_stats.npz and stored in each checkpoint.
    Caveat: x_det is from a model trained on these same years, so the training
    residuals may be smaller than out-of-sample ones; check the spread-skill
    ratio on the test years.
'''

import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator
from torch_ema import ExponentialMovingAverage as EMA
import torch.distributed as dist

from wavediffusion.model_unet import myUnet, DetUnet
from wavediffusion.model import Scaled, ResidualDiffusion
from wavediffusion.wavedata import npyDataWndHist
from wavediffusion.diffusion import ScheduleLogLinear, ScheduleDDPM, masked_training_loop

from wavediffusion.waveutils import evaluate, _save_sample_plots
OPTION = 3
HIST_DURATION_DAYS = 10
HIST_FREQ_DAYS = 1
VAR_NAMES = ['hs_p1', 'tp_p1', 'thetap_p1', 'hs_p2', 'tp_p2', 'thetap_p2', 'ci']
CIRCULAR = [VAR_NAMES.index('thetap_p1'), VAR_NAMES.index('thetap_p2')]
DET_DIR = '/global/homes/j/jiarongw/scratch_folder/review/OPTION3_periodic_hist10d_det/'
DET_CKPT = 20
N_RSTD_SAMPLES = 400


def build_det(train, precond_ch):
    ''' The frozen deterministic model, with its EMA weights. '''
    det = DetUnet(in_dim=320, in_ch=7, out_ch=7, ch=256, precond_ch=precond_ch,
                  scale=(train.meanx, train.stdx, train.meanf, train.stdf),
                  ch_mult=(1, 2, 2), attn_resolutions=(16,), periodic=True)
    ckpt = torch.load(os.path.join(DET_DIR, f'ckpt_{DET_CKPT}.pt'), map_location='cpu')
    det.load_state_dict(ckpt['model'])
    det_ema = EMA(det.parameters(), decay=0.999)
    det_ema.load_state_dict(ckpt['ema'])
    det_ema.copy_to(det.parameters())
    return det.eval()


@torch.no_grad()
def residual_std(model, dataset, device, n_samples, batch_size=8):
    ''' Per-channel std of the (unscaled) residual x - x_det over ice-free pixels
        of n_samples random training samples (fixed seed). '''
    idx = np.random.default_rng(0).choice(len(dataset), size=n_samples, replace=False)
    loader = DataLoader(Subset(dataset, idx), batch_size=batch_size)
    s2, n = 0, 0
    for x, f, mask in loader:
        x, f, mask = x.to(device), f.to(device), mask.to(device)
        _, x_det = model.make_cond(f)
        d = model.raw_residual(x, x_det)
        s2 = s2 + (d ** 2 * mask).sum(dim=(0, 2, 3))
        n = n + mask.sum(dim=(0, 2, 3))
    return (s2 / n).sqrt().cpu().numpy()


@torch.no_grad()
def sample_and_save_resid(model, ema, loader, schedule, accelerator, path, test, filename):
    ''' Plots of one residual-model sample and of x_det for a test batch. '''
    x, f, mask = next(iter(loader))
    x, f, mask = x.to(accelerator.device), f.to(accelerator.device), mask.to(accelerator.device)
    with ema.average_parameters():
        x0, x_det = model.sample(f, mask, schedule.sample_sigmas(40), accelerator=accelerator)
    _save_sample_plots(x, x0, f, mask, test, path, filename, OPTION=OPTION)
    _save_sample_plots(x, x_det, f, mask, test, path, filename + '_det', OPTION=OPTION)


def main(path, train_batch_size=1024, epochs=300, sample_batch_size=64, RESUME=False, weights_file=None,
         ckpt_everyn_epoch=1, sample_everyn_epoch=1, eval_everyn_step=100, gradient_accumulation_steps=4):
    # Setup
    print(torch.cuda.is_available())
    a = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=gradient_accumulation_steps)
    print(a.state)

    train_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/partition_global/'
    forcing_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/mean_global/'
    years = list(range(1993, 2018))
    train_file_names = [*( (f'waveparts_{y:04d}{m:02d}', f'forcing_{y:04d}{m:02d}') for y in years for m in range(1, 13) )]
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
    test_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/partition_global/'
    # Test set: all of 2019 (held out from training, 1993-2017), every TEST_STRIDE-th sample (see loader_test)
    test_file_names = [(f'waveparts_2019{m:02d}', f'forcing_2019{m:02d}') for m in range(1, 13)]
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
    schedule_train = ScheduleDDPM()

    # precond_ch: forcing channels (4 base + 2 per historical wind sample) + the 7 x_det channels
    n_forcing = 4 + 2 * (HIST_DURATION_DAYS // HIST_FREQ_DAYS)
    net = Scaled(myUnet)(in_dim=320, in_ch=7, out_ch=7, ch=256, precond_ch=n_forcing + 7,
                         scale=(train.meanx, train.stdx, train.meanf, train.stdf),
                         ch_mult=(1, 2, 2), attn_resolutions=(16,), periodic=True)
    det = build_det(train, n_forcing)

    # Residual scale: computed once (main process), then read by every process.
    rstd_file = path + 'resid_stats.npz'
    if a.is_main_process and not RESUME and not os.path.exists(rstd_file):
        probe = ResidualDiffusion(net, det, rstd=np.ones(7), stdx=stdx, circular=CIRCULAR).to(a.device)
        rstd = residual_std(probe, train, a.device, N_RSTD_SAMPLES)
        np.savez(rstd_file, rstd=rstd, det_dir=DET_DIR, det_ckpt=DET_CKPT)
        print('Residual std per channel:', dict(zip(VAR_NAMES, np.round(rstd, 4))))
    a.wait_for_everyone()    # every process, so none waits on one that skipped the barrier
    rstd = np.load(rstd_file)['rstd'] if os.path.exists(rstd_file) else np.ones(7)    # resume: from the checkpoint
    model = ResidualDiffusion(net, det, rstd=rstd, stdx=stdx, circular=CIRCULAR)

    # Train
    log_file = open(path + "loss_log.txt", "a")
    test_log_file = open(path + "test_loss_log.txt", "a")
    ema = EMA(model.net.parameters(), decay=0.999)    # det is frozen
    start_epoch = 0

    if RESUME and weights_file is not None:
        ckpt = torch.load(weights_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])    # includes det and rstd
        ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt["epoch"]
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
                        sample_and_save_resid(a.unwrap_model(model), ema, loader_test, schedule_infer, a, path,
                                              test=test, filename=f"sample_epoch{last_epoch}")
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

    path = f'/global/homes/j/jiarongw/scratch_folder/review/OPTION{OPTION}_periodic_hist10d_resid/'
    os.makedirs(path, exist_ok=True)
    # First run (2026-10-25): 5 epochs from scratch -> ckpt_1..5
    # main(path, train_batch_size=4, epochs=5, sample_batch_size=2, RESUME=False, weights_file=None,
    #      ckpt_everyn_epoch=1, sample_everyn_epoch=1, gradient_accumulation_steps=4)
    # Resume: 5 more epochs -> ckpt_6..10
    main(path, train_batch_size=4, epochs=5, sample_batch_size=2, RESUME=True,
         weights_file=path+'ckpt_5.pt', ckpt_everyn_epoch=1, sample_everyn_epoch=1, gradient_accumulation_steps=4)
