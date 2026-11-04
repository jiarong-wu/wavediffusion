''' Sampling for the DDPM model. '''

from wavediffusion.model_unet import myUnet
from wavediffusion.model import Scaled
from wavediffusion.wavedata import npyDataResized, npyDataWndHist
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage as EMA
from wavediffusion.diffusion import ScheduleLogLinear, ScheduleDDPM, samples, samples_thres
from wavediffusion.timeindex import (hist_window, months_file_list, build_date_index, select_sample_indices,
                                     write_dates_csv, to_dataset_idx)
from accelerate import Accelerator
import contextlib
import torch
import time
import numpy as np
import os
from torchvision import transforms as tf
from tqdm import tqdm

############### monthly ##################
OPTION = 1
label = '_periodic'
HIST = True
# Wind-history window; must match how the checkpoint was trained (it sets the
# number of history channels, precond_ch = 4 + 2 * n_hist). Also sets how far
# back the extra month loaded before the first test month must reach.
HIST_DURATION_DAYS = 5
HIST_FREQ_DAYS = 1
MONTHLY = True
test_year = 2004
test_month = 4
every_n = 8
# Optionally pick the dates instead of striding: SAMPLE_DAYS are days of every
# test month (days past a month's end are skipped), at SAMPLE_HOUR (UTC, multiple
# of 3), e.g. SAMPLE_DAYS = [1, 8, 15, 22]. None = every every_n steps from the
# first date with a full history window. Output files are named by dataset index;
# the index -> date mapping is saved to {save_path}dates.csv.
SAMPLE_DAYS = None
SAMPLE_HOUR = 0

# OPTION = 2
# label = '_nohist'
# label = ''
# HIST = True
# MONTHLY = True
# test_year = 2004
# test_month = 9

# OPTION = 3
# label = '_continuous'
# HIST = True
# MONTHLY = True
# test_year = 2004
# test_month = 4
# every_n = 8 # Sampling time interval, 8 is daily, 40 is 5-day interval

n_ensem = 1
epoch = 6
sigma0 = 100
steps = 10
MIXED_PRECISION = False
model_path = f'/global/homes/j/jiarongw/scratch_folder/final/OPTION{OPTION}{label}/'
save_path = f'/global/homes/j/jiarongw/scratch_folder/final/temp/OPTION{OPTION}{label}_sigma{sigma0}_epoch{epoch}_{test_year}{test_month:02d}/'
os.makedirs(save_path, exist_ok=True)

############### A whole year ##################
# OPTION = 3
# label = '_moredata'
# label = '_nohist'
# label = '_continuous'
# HIST = True
# MONTHLY = False
# test_year = 2004
# every_n = 40 # Sampling time interval, 8 is daily, 40 is 5-day interval

# n_ensem = 20
# epoch = 6
# sigma0 = 100
# steps = 40
# MIXED_PRECISION = False
# model_path = f'/global/homes/j/jiarongw/scratch_folder/final/OPTION{OPTION}{label}/'
# save_path = f'/global/homes/j/jiarongw/scratch_folder/final/temp/OPTION{OPTION}{label}_sigma{sigma0}_epoch{epoch}_{test_year}/'
# os.makedirs(save_path, exist_ok=True)

if OPTION == 1:
    var_names = ['hs', 'tm', 'thetam']
elif OPTION == 2:
    var_names = ['hs', 'uss', 'vss', 'mssd', 'mssc']
elif OPTION == 3:
    var_names = ['hs_p1', 'tp_p1', 'thetap_p1', 'hs_p2', 'tp_p2', 'thetap_p2', 'ci']

############## Define testing data ################
forcing_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/mean_global/'
if OPTION == 1 or OPTION == 2:
    test_file_path, x_prefix = forcing_file_path, 'wavemean'
elif OPTION == 3:
    test_file_path, x_prefix = '/global/homes/j/jiarongw/scratch_folder/wave_data/partition_global/', 'waveparts'
test_months = [(test_year, test_month)] if MONTHLY else [(test_year, month) for month in range(1, 13)]
# With wind history, the month before the first test month is loaded too, only so
# the first test days have a full history window; nothing in it is sampled.
test_file_list, load_months = months_file_list(test_months, test_file_path, forcing_file_path,
                                               x_prefix=x_prefix, prepend_previous=HIST)
stats_file = os.path.join(test_file_path, f'stats_OPTION{OPTION}.npz')
stats = np.load(stats_file)
meanx, stdx = stats['meanx'], stats['stdx']
meanf, stdf = stats['meanf'], stats['stdf']
if HIST:
    test = npyDataWndHist(
        test_file_list,
        resize_x=(320,320), resize_f=(320,320), 
        landmaskname=os.path.join(test_file_path, 'mask.npy'),
        use_icymask=True, compute_stats=False,
        meanx=meanx, stdx=stdx, meanf=meanf, stdf=stdf,
        OPTION=OPTION, hist_duration_days=HIST_DURATION_DAYS, hist_freq_days=HIST_FREQ_DAYS
    )
else:
    test = npyDataResized(
        test_file_list,
        resize_x=(320,320), resize_f=(320,320), 
        landmaskname=os.path.join(test_file_path, 'mask.npy'),
        use_icymask=True, compute_stats=False,
        meanx=meanx, stdx=stdx, meanf=meanf, stdf=stdf,
        OPTION=OPTION
    )
    
############### Define loaded model ###############
RESUME = True 
GUIDED = False # If apply guidance during sampling
USING_PRE = False # If using previous mean as initialization for sampling

weights_file = model_path + f'ckpt_{epoch}.pt'
precond_ch = 4 + 2 * hist_window(HIST_DURATION_DAYS, HIST_FREQ_DAYS)[0] if HIST else 4    # base forcing + (u, v) per snapshot
if OPTION == 1:
    model = Scaled(myUnet)(in_dim=320, in_ch=3, out_ch=3, ch=256, precond_ch=precond_ch,
                       scale=(test.meanx, test.stdx, test.meanf, test.stdf),
                       ch_mult=(1, 2, 2), attn_resolutions=(16,), periodic=True)
elif OPTION == 2:
    if HIST:
        model = Scaled(myUnet)(in_dim=320, in_ch=5, out_ch=5, ch=256, precond_ch=precond_ch, 
                           scale=(test.meanx, test.stdx, test.meanf, test.stdf),
                           ch_mult=(1, 2, 2), attn_resolutions=(16,))
    else:   
        model = Scaled(myUnet)(in_dim=320, in_ch=5, out_ch=5, ch=256, precond_ch=precond_ch, 
                        scale=(test.meanx, test.stdx, test.meanf, test.stdf),
                        ch_mult=(1, 2, 2), attn_resolutions=(16,))
elif OPTION == 3:
    model = Scaled(myUnet)(in_dim=320, in_ch=7, out_ch=7, ch=256, precond_ch=precond_ch, 
                       scale=(test.meanx, test.stdx, test.meanf, test.stdf),
                       ch_mult=(1, 2, 2), attn_resolutions=(16,))

ema = EMA(model.parameters(), decay=0.999)
if RESUME:
    ckpt = torch.load(weights_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
a = Accelerator()
model = a.prepare(model)
ema.to(a.device)

############## Define sampling parameters ###########
schedule_infer = ScheduleLogLinear(sigma_min=0.01, sigma_max=sigma0, N=80)
# schedule_infer = ScheduleDDPM()

############# Sampling function ################
# sampling from index sample of test set batched (by stacking f)
# x is the true image (1*channel*320*320)
# xt is the optional initialization of noisy image (1*channel*320*320), if not provided, start from pure noise
@torch.no_grad()
def sample (f, x, mask, xt=None, n_ensem=10):
    global a, test, ema, model, schedule_infer, GUIDED 
    f = f.repeat(n_ensem, 1, 1, 1)    
    xt = xt.repeat(n_ensem, 1, 1, 1) * mask.to(a.device) if xt is not None else None
    if xt is None:
        print("Start with no history snapshot.")
    x0_ensem = []
    amp_ctx = torch.autocast(device_type=a.device.type, dtype=torch.float16) if MIXED_PRECISION else contextlib.nullcontext()
    with ema.average_parameters(), amp_ctx:
        if GUIDED:
            hs_thres = -test.meanx[0]/test.stdx[0]
            t0 = time.perf_counter()
            *xt, x0 = samples_thres(model, schedule_infer.sample_sigmas(steps), gam=1, mu=0.5, batchsize=n_ensem,
                                    thres=hs_thres, accelerator=a, cond=f, mask=mask, xt=xt)
        else:
            t0 = time.perf_counter()
            *xt, x0 = samples(model, schedule_infer.sample_sigmas(steps), gam=1, mu=0.5, batchsize=n_ensem,
                              accelerator=a, cond=f, mask=mask, xt=xt)
        if a.device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        # This operation hasn't been broadcasted
        for i in range(0, n_ensem):
            x_ = test.invert_x(x0[i])
            x_ = x_ * tf.Resize((320,720))(mask[0].to(x_))
            x0_ensem.append(x_.cpu().numpy())
        t2 = time.perf_counter()
        print(f'[timing] diffusion sampling: {t1-t0:.2f}s  |  postprocess ({n_ensem} members): {t2-t1:.2f}s')
    
    x_truth = test.invert_x(x[0]) * tf.Resize((320,720))(mask[0].to(x[0]))
    x_truth = x_truth.cpu().numpy()
    f_ = test.invert_f(f[0]).cpu().numpy()
    x0_ensem = np.array(x0_ensem)
    
    mean = x0_ensem.mean(axis=0)
    std = x0_ensem.std(axis=0)    
    
    return x_truth, mean, std, x0_ensem[0]

    
mse_vars = {name: [] for name in var_names}
std_vars = {name: [] for name in var_names}

############ Prepare mask and lat weights ############
# Weights computed and saved with wavedata.ipynb
mask = test.landmask == 1 # land mask for computing metrics
wlat = np.load('/global/homes/j/jiarongw/scratch_folder/wave_data/wlat.npy')
# weighted and masked RMSE
def weighted_mse(a, b, mask, w):
    diff2 = (a - b) ** 2
    return np.sum(diff2[mask] * w[mask]) / np.sum(w[mask])
# weighted and masked spread / skill
def weighted_meanvar(std, mask, w):
    var = std ** 2
    return np.sum(var[mask] * w[mask]) / np.sum(w[mask])    

############ Sampling indices and their dates ############
padding = getattr(test, 'padding', 0)    # history window; 0 without wind history
dates, _ = build_date_index(load_months, test.file_lengths)
raw_indices = select_sample_indices(dates, padding, every_n, SAMPLE_DAYS, SAMPLE_HOUR, keep_months=test_months)
indices = [to_dataset_idx(r, padding) for r in raw_indices]
write_dates_csv(f'{save_path}dates.csv', raw_indices, dates, padding)

x_truth, mean, std = None, None, None
for index in tqdm(indices):
# for index in tqdm(range(0, 96, 8)):
    print(f'Sampling for index {index}...')
    x, f, icymask = test.__getitem__(index)
    x = x.unsqueeze(0); f = f.unsqueeze(0); icymask = icymask.unsqueeze(0)
    if index == indices[0]:
        x_truth, x_mean, std, rsample = sample (f, x, icymask, xt=None, n_ensem=n_ensem)
    else:
        if USING_PRE:
            # To make dimension consistent transform mean
            xt = test.transform_x(torch.tensor(mean).to(a.device)) + schedule_infer.sample_sigmas(steps)[0].to(a.device) * torch.randn(model.input_dims).to(a.device)
            x_truth, x_mean, std, rsample = sample (f, x, icymask, xt=xt, n_ensem=n_ensem)
        else:
            x_truth, x_mean, std, rsample = sample (f, x, icymask, xt=None, n_ensem=n_ensem)
    f_ = test.invert_f(f[0]).cpu().numpy()
    # Lower bound the wave period with 0
    # if OPTION == 1:
    #     x_mean[1] = np.maximum(x_mean[1], 0)
    #     x_truth[1] = np.maximum(x_truth[1], 0)
    # elif OPTION == 3:
    #     x_mean[1] = np.maximum(x_mean[1], 0)
    #     x_truth[1] = np.maximum(x_truth[1], 0)  
    #     x_mean[4] = np.maximum(x_mean[4], 0)
    #     x_truth[4] = np.maximum(x_truth[4], 0)      
    # Resize icymask from model resolution (320x320) to output resolution (320x720)
    icymask_resized = tf.Resize((320, 720))(icymask[0].float()).numpy()[0] > 0.9

    np.save(f'{save_path}sample_{index}.npy', rsample)
    np.save(f'{save_path}mean_{index}.npy', x_mean)
    np.save(f'{save_path}std_{index}.npy', std)
    np.save(f'{save_path}truth_{index}.npy', x_truth)
    np.save(f'{save_path}forcing_{index}.npy', f_[0:4]) # wind u, v, depth, and ice
    np.save(f'{save_path}icymask_{index}.npy', icymask_resized)

    # Use the ice mask?
    for name in var_names:
         mse_vars[name].append(weighted_mse(x_truth[var_names.index(name)], x_mean[var_names.index(name)], icymask_resized, wlat))
         std_vars[name].append(weighted_meanvar(std[var_names.index(name)], icymask_resized, wlat))

for name in var_names:
    mse_vars[name] = np.array(mse_vars[name])
    std_vars[name] = np.array(std_vars[name])
    print(f'{name} mse: {mse_vars[name].mean()**0.5:.4f} \\pm {np.std(mse_vars[name]**0.5):.4f}')
    print(f'{name} ssr: {(std_vars[name].mean() / mse_vars[name].mean())**0.5:.4f}')    