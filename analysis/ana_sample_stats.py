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
# test_file_names = [*( (f'wave_2004{i:02d}', f'forcing_2004{i:02d}') for i in range(1, 13) )]
test_file_names = [('wave_200409', 'forcing_200409')]
test_file_list = [(os.path.join(test_file_path, f'{x}.npy'), 
                   os.path.join(test_file_path, f'{f}.npy')) for x, f in test_file_names]
stats_file = os.path.join(train_file_path, 'stats.npz')
stats = np.load(stats_file)
meanx, stdx = stats['meanx'], stats['stdx']
meanf, stdf = stats['meanf'], stats['stdf']
test = npyDataWndHist(
    test_file_list,
    resize_x=(320,320), resize_f=(320,320), 
    landmaskname=os.path.join(test_file_path, 'mask.npy'),
    use_icymask=True, compute_stats=False,
    meanx=meanx, stdx=stdx, meanf=meanf, stdf=stdf
)

############### Define loaded model ###############
RESUME = True 
GUIDED = False # If apply guidance during sampling
USING_PRE = False # If using previous mean as initialization for sampling
n_ensem = 20
epoch = 16
path = '/global/homes/j/jiarongw/scratch_folder/log1p/hist1/'
weights_file = path + f'ckpt_{epoch}.pt'
model = Scaled(myUnet)(in_dim=320, in_ch=4, out_ch=4, ch=256, precond_ch=13, 
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
schedule_infer = ScheduleLogLinear(sigma_min=0.01, sigma_max=80, N=80)


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

savepath = '/global/homes/j/jiarongw/wavediffusion/example/temp/200409_DDPM_40steps_100_20ensem_0.5d/'

hs_mse, lp_mse, tp_mse, thetap_mse, spread_mse = [], [], [], [], []
hs_var, lp_var, thetap_var, spread_var = [], [], [], []

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
for index in tqdm(range(0, test.__len__(), 4)):
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
    x_mean[1] = np.maximum(x_mean[1], 0)
    x_truth[1] = np.maximum(x_truth[1], 0)     
    
    np.save(f'{savepath}sample_{index}.npy', rsample)
    np.save(f'{savepath}mean_{index}.npy', x_mean)
    np.save(f'{savepath}std_{index}.npy', std)
    np.save(f'{savepath}truth_{index}.npy', x_truth)
    np.save(f'{savepath}forcing_{index}.npy', f_[0:3]) # wind u, v, and ice concentration
    # Use the ice mask?
    icymask = f_[2] == 1
    hs_mse.append(weighted_mse(x_truth[0], x_mean[0], icymask, wlat))
    lp_mse.append(weighted_mse(x_truth[1], x_mean[1], icymask, wlat))
    tp_mse.append(weighted_mse((x_truth[1]/1.56)**0.5, (x_mean[1]/1.56)**0.5, icymask, wlat))
    thetap_mse.append(weighted_mse(x_truth[2], x_mean[2], icymask, wlat))
    spread_mse.append(weighted_mse(x_truth[3], x_mean[3], icymask, wlat))
    hs_var.append(weighted_meanvar(std[0], icymask, wlat))
    lp_var.append(weighted_meanvar(std[1], icymask, wlat))
    thetap_var.append(weighted_meanvar(std[2], icymask, wlat))
    spread_var.append(weighted_meanvar(std[3], icymask, wlat))  
      
hs_mse = np.array(hs_mse); lp_mse = np.array(lp_mse); tp_mse = np.array(tp_mse); thetap_mse = np.array(thetap_mse) ; spread_mse = np.array(spread_mse) 
hs_var = np.array(hs_var); lp_var = np.array(lp_var); thetap_var = np.array(thetap_var); spread_var = np.array(spread_var)

print(f'hs mse: {hs_mse.mean()**0.5:.2f} \\pm {np.std(hs_mse**0.5):.2f}')
print(f'lp mse: {lp_mse.mean()**0.5:.2f} \\pm {np.std(lp_mse**0.5):.2f}')
print(f'tp mse: {tp_mse.mean()**0.5:.2f} \\pm {np.std(tp_mse**0.5):.2f}')
print(f'thetap mse: {thetap_mse.mean()**0.5:.2f} \\pm {np.std(thetap_mse**0.5):.2f}')
print(f'spread mse: {spread_mse.mean()**0.5:.2f} \\pm {np.std(spread_mse**0.5):.2f}')

print(f'hs ssr: {(hs_var.mean() / hs_mse.mean())**0.5:.2f}')
print(f'lp ssr: {(lp_var.mean() / lp_mse.mean())**0.5:.2f}')
print(f'thetap ssr: {(thetap_var.mean() / thetap_mse.mean())**0.5:.2f}')
print(f'spread ssr: {(spread_var.mean() / spread_mse.mean())**0.5:.2f}')   