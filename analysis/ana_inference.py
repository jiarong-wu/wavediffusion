''' Inference for the simple deterministic model. '''

from wavediffusion.model import Scaled
from wavediffusion.wavedata import npyDataWndHist
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage as EMA
from wavediffusion.plain_unet import plainUnet, masked_training_loop_plain, evaluate_plain
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
test_file_names = [('wave_200412', 'forcing_200412')]
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
epoch = 4
path = '/global/homes/j/jiarongw/scratch_folder/log1p/plain_hist/'
weights_file = path + f'ckpt_{epoch}.pt'
model = plainUnet(in_dim=320, in_ch=13, out_ch=4, ch=256, 
                    ch_mult=(1, 2, 2), attn_resolutions=(16,)) 

ema = EMA(model.parameters(), decay=0.999)
if RESUME:
    ckpt = torch.load(weights_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
a = Accelerator()
ema, model = a.prepare(ema, model)

# Snapshot saving path
savepath = '/global/homes/j/jiarongw/wavediffusion/example/temp/plain_200412/'
hs_mse, lp_mse, tp_mse, thetap_mse, spread_mse = [], [], [], [], []  

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
    
with torch.no_grad():
    for index in tqdm(range(0, test.__len__(), 4)):
        print(f'Inference for index {index}...')
        x, f, icymask = test.__getitem__(index)
        x = x.unsqueeze(0).to(a.device); f = f.unsqueeze(0).to(a.device); icymask = icymask.unsqueeze(0).to(a.device)
        with ema.average_parameters():
            y = model(f)
            x_ = test.invert_x(y[0])
            x_pred = x_ * tf.Resize((320,720))(icymask[0].to(x_)); x_pred = x_pred.cpu().numpy()
            x_truth = test.invert_x(x[0]) * tf.Resize((320,720))(icymask[0].to(x)); x_truth = x_truth.cpu().numpy()
            f_ = test.invert_f(f[0]).cpu().numpy()
            # Lower bound the wave length with 0?
            x_pred[1] = np.maximum(x_pred[1], 0)
            x_truth[1] = np.maximum(x_truth[1], 0)            
            
        np.save(f'{savepath}pred_{index}.npy', x_pred)
        np.save(f'{savepath}truth_{index}.npy', x_truth)
        np.save(f'{savepath}forcing_{index}.npy', f_[0:3]) # wind u, v, and ice concentration
        # Use the ice mask
        icymask = f_[2] == 1
        hs_mse.append(weighted_mse(x_truth[0], x_pred[0], icymask, wlat))
        lp_mse.append(weighted_mse(x_truth[1], x_pred[1], icymask, wlat))
        tp_mse.append(weighted_mse((x_truth[1]/1.56)**0.5, (x_pred[1]/1.56)**0.5, icymask, wlat))
        thetap_mse.append(weighted_mse(x_truth[2], x_pred[2], icymask, wlat))
        spread_mse.append(weighted_mse(x_truth[3], x_pred[3], icymask, wlat))
        
hs_mse = np.array(hs_mse); lp_mse = np.array(lp_mse); tp_mse = np.array(tp_mse); thetap_mse = np.array(thetap_mse) ; spread_mse = np.array(spread_mse) 

print(f'hs mse: {hs_mse.mean()**0.5:.2f} \\pm {np.std(hs_mse**0.5):.2f}')
print(f'lp mse: {lp_mse.mean()**0.5:.2f} \\pm {np.std(lp_mse**0.5):.2f}')
print(f'tp mse: {tp_mse.mean()**0.5:.2f} \\pm {np.std(tp_mse**0.5):.2f}')
print(f'thetap mse: {thetap_mse.mean()**0.5:.2f} \\pm {np.std(thetap_mse**0.5):.2f}')
print(f'spread mse: {spread_mse.mean()**0.5:.2f} \\pm {np.std(spread_mse**0.5):.2f}')
