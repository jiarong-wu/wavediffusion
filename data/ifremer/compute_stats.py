import os
import numpy as np
from wavediffusion.wavedata import npyDataResized

# --- Paths and file lists ---
train_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/train_global/'
train_file_names = [(f'waveparts_2011{i:02d}', f'forcing_2011{i:02d}') for i in range(1, 13)] + \
                   [(f'waveparts_2012{i:02d}', f'forcing_2012{i:02d}') for i in range(1, 13)] + \
                   [(f'waveparts_2013{i:02d}', f'forcing_2013{i:02d}') for i in range(1, 13)] + \
                   [(f'waveparts_2014{i:02d}', f'forcing_2014{i:02d}') for i in range(1, 13)] + \
                   [(f'waveparts_2015{i:02d}', f'forcing_2015{i:02d}') for i in range(1, 13)]

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
    OPTION=3 
)

# --- Save stats to disk ---
# stats_file = os.path.join(train_file_path, 'stats.npz')
# np.savez(stats_file,
#          meanx=dataset.meanx,
#          stdx=dataset.stdx,
#          meanf=dataset.meanf,
#          stdf=dataset.stdf)
stats_file = os.path.join(train_file_path, 'stats_parts.npz')
dataset.meanx[-1] = 0 # Set mean of crossing sea criterion to 0 
dataset.stdx[-1] = 1 # Set std of crossing sea criterion to 1 
np.savez(stats_file,
         meanparts=dataset.meanx,
         stdparts=dataset.stdx)
print(f'Mean {dataset.meanx}')
print(f'Std {dataset.stdx}')
print(f"Stats saved to {stats_file}")



