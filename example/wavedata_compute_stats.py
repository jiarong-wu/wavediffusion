import os
import numpy as np
from smalldiffusion.wavedata import npyDataResized

# --- Paths and file lists ---
train_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/stokes_global/'
train_file_names = [(f'wave_2011{i:02d}', f'forcing_2011{i:02d}') for i in range(1, 13)] + \
                   [(f'wave_2012{i:02d}', f'forcing_2012{i:02d}') for i in range(1, 13)] + \
                   [(f'wave_2013{i:02d}', f'forcing_2013{i:02d}') for i in range(1, 13)] + \
                   [(f'wave_2014{i:02d}', f'forcing_2014{i:02d}') for i in range(1, 13)] + \
                   [(f'wave_2015{i:02d}', f'forcing_2015{i:02d}') for i in range(1, 13)]

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
    compute_stats=True
)

# --- Save stats to disk ---
stats_file = os.path.join(train_file_path, 'stats.npz')
np.savez(stats_file,
         meanx=dataset.meanx,
         stdx=dataset.stdx,
         meanf=dataset.meanf,
         stdf=dataset.stdf)

print(f"Stats saved to {stats_file}")