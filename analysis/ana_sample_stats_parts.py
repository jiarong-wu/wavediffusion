''' Sampling for the DDPM model. '''

from wavediffusion.model_unet import myUnet
from wavediffusion.model import Scaled
from wavediffusion.wavedata import npyDataResized, npyDataWndHist, npyDataWndHistDaily
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage as EMA
from wavediffusion.diffusion import ScheduleLogLinear, ScheduleDDPM, samples, samples_thres
from accelerate import Accelerator
import torch
import numpy as np
import os
from torchvision import transforms as tf
from tqdm import tqdm

############## Define testing data ################
train_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/train_global/'
test_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/test_global/'
test_file_names = [*( (f'waveparts_2004{i:02d}', f'forcing_2004{i:02d}') for i in range(1, 13) )]
# test_file_names = [('waveparts_200409', 'forcing_200409')]
test_file_list = [(os.path.join(test_file_path, f'{x}.npy'), 
                   os.path.join(test_file_path, f'{f}.npy')) for x, f in test_file_names]
stats_file1 = os.path.join(train_file_path, 'stats.npz')
stats = np.load(stats_file1)
meanf, stdf = stats['meanf'], stats['stdf']
stats_file2 = os.path.join(train_file_path, 'stats_parts.npz')
stats_parts = np.load(stats_file2)
meanx, stdx = stats_parts['meanparts'], stats_parts['stdparts']
test = npyDataWndHist(
    test_file_list,
    resize_x=(320,320), resize_f=(320,320), 
    landmaskname=os.path.join(test_file_path, 'mask.npy'),
    use_icymask=True, compute_stats=False,
    meanx=meanx, stdx=stdx, meanf=meanf, stdf=stdf,
    OPTION=3
)

############### Define loaded model ###############
RESUME = True 
GUIDED = False # If apply guidance during sampling
USING_PRE = False # If using previous mean as initialization for sampling
n_ensem = 20
epoch = 32
path = '/global/homes/j/jiarongw/scratch_folder/log1p/waveparts_hist/'
weights_file = path + f'ckpt_{epoch}.pt'
model = Scaled(myUnet)(in_dim=320, in_ch=7, out_ch=7, ch=256, precond_ch=13, 
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
schedule_infer = ScheduleLogLinear(sigma_min=0.01, sigma_max=100, N=80)


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
    with ema.average_parameters():
        if GUIDED:
            hs_thres = -test.meanx[0]/test.stdx[0]
            *xt, x0 = samples_thres(model, schedule_infer.sample_sigmas(40), gam=1, mu=0.5, batchsize=n_ensem, 
                                    thres=hs_thres, accelerator=a, cond=f, mask=mask, xt=xt)
        else:
            *xt, x0 = samples(model, schedule_infer.sample_sigmas(40), gam=1, mu=0.5, batchsize=n_ensem,
                              accelerator=a, cond=f, mask=mask, xt=xt)
        # This operation hasn't been broadcasted
        for i in range(0, n_ensem):
            x_ = test.invert_x(x0[i])
            x_ = x_ * tf.Resize((320,720))(mask[0].to(x_))
            x0_ensem.append(x_.cpu().numpy())
    
    x_truth = test.invert_x(x[0]) * tf.Resize((320,720))(mask[0].to(x))
    x_truth = x_truth.cpu().numpy()
    f_ = test.invert_f(f[0]).cpu().numpy()
    x0_ensem = np.array(x0_ensem)
    
    mean = x0_ensem.mean(axis=0)
    std = x0_ensem.std(axis=0)    
    
    return x_truth, mean, std, x0_ensem[0]

savepath = path + '2004/'
os.makedirs(savepath, exist_ok=True)

hs_part1_mse, tp_part1_mse, thetap_part1_mse, hs_part2_mse, tp_part2_mse, thetap_part2_mse = [], [], [], [], [], []
hs_part1_var, tp_part1_var, thetap_part1_var, hs_part2_var, tp_part2_var, thetap_part2_var = [], [], [], [], [], []

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

x_truth, mean, std = None, None, None
for index in tqdm(range(0, test.__len__(), 80)):
    print(f'Sampling for index {index}...')
    x, f, icymask = test.__getitem__(index)
    x = x.unsqueeze(0); f = f.unsqueeze(0); icymask = icymask.unsqueeze(0)
    if index == 0:
        x_truth, x_mean, std, rsample = sample (f, x, icymask, xt=None, n_ensem=n_ensem)
    else:
        if USING_PRE:
            # To make dimension consistent transform mean
            xt = test.transform_x(torch.tensor(mean).to(a.device)) + schedule_infer.sample_sigmas(40)[0].to(a.device) * torch.randn(model.input_dims).to(a.device)
            x_truth, x_mean, std, rsample = sample (f, x, icymask, xt=xt, n_ensem=n_ensem)
        else:
            x_truth, x_mean, std, rsample = sample (f, x, icymask, xt=None, n_ensem=n_ensem)
    f_ = test.invert_f(f[0]).cpu().numpy()
    # Lower bound the wave length with 0?
    # x_mean[1] = np.maximum(x_mean[1], 0)
    # x_truth[1] = np.maximum(x_truth[1], 0)     
    
    np.save(f'{savepath}sample_{index}.npy', rsample)
    np.save(f'{savepath}mean_{index}.npy', x_mean)
    np.save(f'{savepath}std_{index}.npy', std)
    np.save(f'{savepath}truth_{index}.npy', x_truth)
    np.save(f'{savepath}forcing_{index}.npy', f_[0:3]) # wind u, v, and ice concentration
    # Use the ice mask?
    icymask = f_[2] == 1
    hs_part1_mse.append(weighted_mse(x_truth[0], x_mean[0], icymask, wlat))
    tp_part1_mse.append(weighted_mse(x_truth[1], x_mean[1], icymask, wlat))
    thetap_part1_mse.append(weighted_mse(x_truth[2], x_mean[2], icymask, wlat))
    hs_part2_mse.append(weighted_mse(x_truth[3], x_mean[3], icymask, wlat))
    tp_part2_mse.append(weighted_mse(x_truth[4], x_mean[4], icymask, wlat))
    thetap_part2_mse.append(weighted_mse(x_truth[5], x_mean[5], icymask, wlat))
    hs_part1_var.append(weighted_meanvar(std[0], icymask, wlat))
    tp_part1_var.append(weighted_meanvar(std[1], icymask, wlat))
    thetap_part1_var.append(weighted_meanvar(std[2], icymask, wlat))
    hs_part2_var.append(weighted_meanvar(std[3], icymask, wlat))
    tp_part2_var.append(weighted_meanvar(std[4], icymask, wlat))
    thetap_part2_var.append(weighted_meanvar(std[5], icymask, wlat))
      
print(f'hs part 1 rmse: {np.array(hs_part1_mse).mean()**0.5:.2f} \\pm {np.std(np.array(hs_part1_mse)**0.5):.2f}')
print(f'tp part 1 rmse: {np.array(tp_part1_mse).mean()**0.5:.2f} \\pm {np.std(np.array(tp_part1_mse)**0.5):.2f}')
print(f'thetap part 1 rmse: {np.array(thetap_part1_mse).mean()**0.5:.2f} \\pm {np.std(np.array(thetap_part1_mse)**0.5):.2f}')
print(f'hs part 2 rmse: {np.array(hs_part2_mse).mean()**0.5:.2f} \\pm {np.std(np.array(hs_part2_mse)**0.5):.2f}')
print(f'tp part 2 rmse: {np.array(tp_part2_mse).mean()**0.5:.2f} \\pm {np.std(np.array(tp_part2_mse)**0.5):.2f}')
print(f'thetap part 2 rmse: {np.array(thetap_part2_mse).mean()**0.5:.2f} \\pm {np.std(np.array(thetap_part2_mse)**0.5):.2f}')
print(f'hs part1 ssr: {(np.array(hs_part1_var).mean() / np.array(hs_part1_mse).mean())**0.5:.2f}')
print(f'tp part1 ssr: {(np.array(tp_part1_var).mean() / np.array(tp_part1_mse).mean())**0.5:.2f}')
print(f'thetap part1 ssr: {(np.array(thetap_part1_var).mean() / np.array(thetap_part1_mse).mean())**0.5:.2f}')
print(f'hs part2 ssr: {(np.array(hs_part2_var).mean() / np.array(hs_part2_mse).mean())**0.5:.2f}')
print(f'tp part2 ssr: {(np.array(tp_part2_var).mean() / np.array(tp_part2_mse).mean())**0.5:.2f}')
print(f'thetap part2 ssr: {(np.array(thetap_part2_var).mean() / np.array(thetap_part2_mse).mean())**0.5:.2f}')