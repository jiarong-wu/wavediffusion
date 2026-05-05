''' Compute the mean and std of the training dataset. 
    OPTION = 1: mean for (hs, tm, thetam)
    OPTION = 2: mean for (hs, uss, vss)
    OPTION = 3: mean for (hs_p1, t_p1, theta_p1, hs_p2, t_p2, theta_p2, crossing_sea_criterion)
'''
OPTION = 2
import os
import numpy as np
from wavediffusion.wavedata import npyDataResized

# --- Paths and file lists ---
if OPTION == 3:
    train_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/partition_global/'
    train_file_names = [(f'waveparts_2011{i:02d}', f'forcing_2011{i:02d}') for i in range(1, 13)] + \
                    [(f'waveparts_2012{i:02d}', f'forcing_2012{i:02d}') for i in range(1, 13)] + \
                    [(f'waveparts_2013{i:02d}', f'forcing_2013{i:02d}') for i in range(1, 13)] + \
                    [(f'waveparts_2014{i:02d}', f'forcing_2014{i:02d}') for i in range(1, 13)] + \
                    [(f'waveparts_2015{i:02d}', f'forcing_2015{i:02d}') for i in range(1, 13)]
elif OPTION == 1 or OPTION == 2:
    train_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/mean_global/'
    train_file_names = [(f'wavemean_2011{i:02d}', f'forcing_2011{i:02d}') for i in range(1, 13)] + \
                    [(f'wavemean_2012{i:02d}', f'forcing_2012{i:02d}') for i in range(1, 13)] + \
                    [(f'wavemean_2013{i:02d}', f'forcing_2013{i:02d}') for i in range(1, 13)] + \
                    [(f'wavemean_2014{i:02d}', f'forcing_2014{i:02d}') for i in range(1, 13)] + \
                    [(f'wavemean_2015{i:02d}', f'forcing_2015{i:02d}') for i in range(1, 13)]
    train_file_names = [(f'wavemean_2011{i:02d}', f'forcing_2011{i:02d}') for i in range(1, 13)] 

train_file_list = [(os.path.join(train_file_path, f'{x}.npy'), 
                    os.path.join(train_file_path, f'{f}.npy')) for x, f in train_file_names]

# --- Construct dataset and compute stats ---
print("Computing dataset statistics... This may take a while.")
dataset = npyDataResized(
    train_file_list,
    resize_x=(320,320),
    resize_f=(320,320),
    landmaskname=os.path.join(train_file_path, 'mask.npy'),
    use_icymask=True,
    compute_stats=True,
    OPTION=OPTION
)

# --- Save stats to disk ---
stats_file = os.path.join(train_file_path, f'stats_OPTION{OPTION}.npz')
np.savez(stats_file,
         meanx=dataset.meanx,
         stdx=dataset.stdx,
         meanf=dataset.meanf,
         stdf=dataset.stdf)

print(f'Mean x {dataset.meanx}')
print(f'Std x {dataset.stdx}')
print(f'Mean f {dataset.meanf}')
print(f'Std f {dataset.stdf}')
print(f"Stats saved to {stats_file}")



