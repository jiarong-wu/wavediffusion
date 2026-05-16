''' Sampling for the DDPM model. '''

from wavediffusion.model_unet import myUnet
from wavediffusion.model import Scaled
from wavediffusion.wavedata import npyDataResized, npyDataWndHist
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage as EMA
from wavediffusion.diffusion import ScheduleLogLinear, ScheduleDDPM, samples, samples_thres
from accelerate import Accelerator
import torch
import numpy as np
import os
from torchvision import transforms as tf
from tqdm import tqdm

OPTION = 3
HIST = True
test_year = 2004
test_month = 4

n_ensem = 20
epoch = 12
sigma0 = 100
model_path = f'/global/homes/j/jiarongw/scratch_folder/final/OPTION{OPTION}/'
save_path = f'/global/homes/j/jiarongw/scratch_folder/final/temp/OPTION{OPTION}_sigma{sigma0}_epoch{epoch}_{test_year}{test_month:02d}/'
save_path = f'/global/homes/j/jiarongw/scratch_folder/final/temp/OPTION{OPTION}_DDPM_epoch{epoch}_{test_year}{test_month:02d}/'
os.makedirs(save_path, exist_ok=True)

if OPTION == 1:
    var_names = ['hs', 'tp', 'thetap']
elif OPTION == 2:
    var_names = ['hs', 'uss', 'vss', 'mssd', 'mssc']
elif OPTION == 3:
    var_names = ['hs_p1', 'tp_p1', 'thetap_p1', 'hs_p2', 'tp_p2', 'thetap_p2', 'ci']

############## Define testing data ################
if OPTION == 1 or OPTION == 2:
    test_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/mean_global/'
    # test_file_names = [*( (f'wave_2004{i:02d}', f'forcing_2004{i:02d}') for i in range(1, 13) )]
    test_file_names = [('wavemean_200404', 'forcing_200404')]
    test_file_list = [(os.path.join(test_file_path, f'{x}.npy'), 
                       os.path.join(test_file_path, f'{f}.npy')) for x, f in test_file_names]
elif OPTION == 3:
    test_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/partition_global/'
    forcing_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/mean_global/'
    test_file_names = [('waveparts_200404', 'forcing_200404')]
    test_file_list = [(os.path.join(test_file_path, f'{x}.npy'), 
                       os.path.join(forcing_file_path, f'{f}.npy')) for x, f in test_file_names]
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
        OPTION=OPTION
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
if OPTION == 1:
    model = Scaled(myUnet)(in_dim=320, in_ch=3, out_ch=3, ch=256, precond_ch=14, 
                       scale=(test.meanx, test.stdx, test.meanf, test.stdf),
                       ch_mult=(1, 2, 2), attn_resolutions=(16,))
elif OPTION == 2:
    if HIST:
        model = Scaled(myUnet)(in_dim=320, in_ch=5, out_ch=5, ch=256, precond_ch=14, 
                           scale=(test.meanx, test.stdx, test.meanf, test.stdf),
                           ch_mult=(1, 2, 2), attn_resolutions=(16,))
    else:   
        model = Scaled(myUnet)(in_dim=320, in_ch=5, out_ch=5, ch=256, precond_ch=4, 
                        scale=(test.meanx, test.stdx, test.meanf, test.stdf),
                        ch_mult=(1, 2, 2), attn_resolutions=(16,))
elif OPTION == 3:
    model = Scaled(myUnet)(in_dim=320, in_ch=7, out_ch=7, ch=256, precond_ch=14, 
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
# schedule_infer = ScheduleLogLinear(sigma_min=0.01, sigma_max=sigma0, N=80)
schedule_infer = ScheduleDDPM()

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

x_truth, mean, std = None, None, None
for index in tqdm(range(0, test.__len__(), 8)):
# for index in tqdm(range(0, 96, 8)):
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
    icymask_resized = tf.Resize((320, 720))(icymask[0].float()).numpy()[0] == 1

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