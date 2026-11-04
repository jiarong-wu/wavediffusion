import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tf
from pathlib import Path
from tqdm import tqdm

''' Data loading and processing classes for wave diffusion model.
    wave mean files have 9 channels: hs, t0m1, dir, spr, uuss, vuss, mssu, mssc, mssd
        OPTION 1: use [0,1,2] (hs, t0m1, dir) as x channels
        OPTION 2: use [0,4,5,6,7] (hs, uuss, vuss, mssu, mssc) as x channels (for now)
    wave partition files have 7 channels: hs_p1, tp_p1, dir_p1, hs_p2, tp_p2, dir_p2, crossing sea criterion
        OPTION 3: use all 7 channels as x channels
    f files have 5 channels: uwnd, vwnd, icymask, dpt, ice fraction (skipping channel 3 when loading, but use as mask)
    For all options, use [0,1,3,4] (uwnd, vwnd, dpt, ice fraction) as f channels.
    hs, hs_p1, hs_p2 are log1p transformed before normalization.
    Within mean and std computation:
        mean and std of icymask and ice fraction set to 0 and 1 respectively (not normalized); 
        mean and std of crossing sea criterion set to 0 and 1 respectively (not normalized).
'''

### Multi file data class 
class MultiFileNpyData(Dataset):
    def __init__(self, 
        file_list,  # List of tuples: [(Xname, Fname), ...] for each month
        landmaskname=None, use_icymask=True,
        compute_stats=True,
        meanx=None, stdx=None, meanf=None, stdf=None, 
        OPTION=1,
    ):
        """
        Args:
            file_list: List of tuples, each containing (X_filepath, F_filepath) for one month
            maskname: Path to mask file (static mask only)
            compute_stats: Whether to compute mean/std from data
            meanx, stdx, meanf, stdf: Precomputed statistics (optional)
        """
        super().__init__()

        # OPTION1: wave height + wave period + direction 
        # OPTION2: wave height + stokes drift u + v
        # OPTION3: partition 1 wave height + wave period + direction + partition 2 wave height + wave period + direction + crossing sea criterion
        self.OPTION = OPTION
        
        # Load all files with memory mapping
        self.X_files = []
        self.F_files = []
        self.file_lengths = []
        self.cumulative_lengths = [0]
        
        for x_path, f_path in file_list:
            x_mmap = np.load(x_path, mmap_mode="r")
            f_mmap = np.load(f_path, mmap_mode="r")
            
            self.X_files.append(x_mmap)
            self.F_files.append(f_mmap)  # keep as mmap; avoids loading all years into RAM at init
            
            file_len = len(x_mmap)
            self.file_lengths.append(file_len)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + file_len)
        
        self.total_length = self.cumulative_lengths[-1]
        
        # Load mask (fixed land and from mask file)
        # Used for data transform
        if landmaskname != None:
            self.landmask = np.load(landmaskname)
        # Optional: use icymask (time dependent and frm F_files) for e.g. loss
        self.use_icymask = use_icymask

        # Compute or load statistics
        if compute_stats:
            print("Computing dataset mean and std across all files...")
            meanx, stdx, meanf, stdf = self._compute_mean_std()
        else:
            assert meanx is not None and stdx is not None, "Must provide meanx/stdx when compute_stats=False"
            assert meanf is not None and stdf is not None, "Must provide meanf/stdf when compute_stats=False"
        
        # Store as tensors
        self.meanx = torch.as_tensor(meanx, dtype=torch.float32)
        self.stdx = torch.as_tensor(stdx, dtype=torch.float32)
        self.meanf = torch.as_tensor(meanf, dtype=torch.float32)
        self.stdf = torch.as_tensor(stdf, dtype=torch.float32)
        
        # Transforms
        self.tf_x = tf.Compose([
            tf.Normalize(self.meanx.tolist(), self.stdx.tolist()),
        ])
        self.tf_f = tf.Compose([    
            tf.Normalize(self.meanf.tolist(), self.stdf.tolist()), 
        ])    
        self.inv_tf_x = tf.Compose([
            tf.Normalize(mean=[0]*len(self.meanx), std=(1/self.stdx).tolist()),
            tf.Normalize(mean=(-(self.meanx/self.stdx)).tolist(), std=[1]*len(self.stdx))
        ])
        self.inv_tf_f = tf.Compose([    
            tf.Normalize(mean=[0]*len(self.meanf), std=(1/self.stdf).tolist()),
            tf.Normalize(mean=(-(self.meanf/self.stdf)).tolist(), std=[1]*len(self.stdf))
        ])

    def _get_file_and_local_idx(self, global_idx):
        """Convert global index to (file_index, local_index) tuple."""
        for file_idx in range(len(self.file_lengths)):
            if global_idx < self.cumulative_lengths[file_idx + 1]:
                local_idx = global_idx - self.cumulative_lengths[file_idx]
                return file_idx, local_idx
        raise IndexError(f"Index {global_idx} out of range")

    def __len__(self):
        return self.total_length
    
    def _compute_mean_std(self):
        """Compute channel-wise mean and std across all files."""  

        # Manually specify each option x channel number                         
        if self.OPTION == 1:
            n_channels_x = 3
        elif self.OPTION == 2:   
            n_channels_x = 5
        elif self.OPTION == 3:
            n_channels_x = 7                    
        n_channels_f = 4  # Only consider uwnd, vwnd, dpt, ice fraction for normalization; skip icymask since it's binary and used as mask.
        
        # Initialize accumulators
        sum_x = np.zeros(n_channels_x)
        sum_f = np.zeros(n_channels_f)
        sum_sq_x = np.zeros(n_channels_x)
        sum_sq_f = np.zeros(n_channels_f)
        total_pixels = 0
        
        # Accumulate statistics from each file
        for x_file, f_file in zip(self.X_files, self.F_files):
            if self.use_icymask:
                mask = f_file[:, 2, :, :]  
                mask_broadcast = mask[:, None, :, :].astype(bool)   
                total_pixels += mask_broadcast.sum()
                # print(total_pixels)
            else:
                mask = self.landmask.astype(bool)
                mask_broadcast = mask[None, None, :, :]
                n_samples = len(x_file)
                n_valid_pixels = mask.sum() * n_samples
                total_pixels += n_valid_pixels
                # print(total_pixels)
           
            # Preprocess x_file: log1p on channel 0                            
            if self.OPTION == 1:
                x_file_proc = x_file.copy()[:,[0,1,2]] # avoid modifying mmap, take only the relevant channels
                x_file_proc[:, 0, :, :] = np.log1p(x_file_proc[:, 0, :, :])
                n_channels_x = 3
            elif self.OPTION == 2:   
                x_file_proc = x_file.copy()[:,[0,4,5,6,7]]
                x_file_proc[:, 0, :, :] = np.log1p(x_file_proc[:, 0, :, :])
                n_channels_x = 5
            elif self.OPTION == 3:
                x_file_proc = x_file.copy()
                x_file_proc[:, 0, :, :] = np.log1p(x_file_proc[:, 0, :, :])
                x_file_proc[:, 3, :, :] = np.log1p(x_file_proc[:, 3, :, :]) 
                n_channels_x = 7
            # Select only wind and depth and ice fraction
            f_file_select = f_file[:, [0,1,3,4], :, :]
                    
            # Apply mask
            X_masked = x_file_proc * mask_broadcast
            F_masked = f_file_select * mask_broadcast
        
            # Accumulate sums
            sum_x += np.nansum(X_masked, axis=(0, 2, 3))
            sum_f += np.nansum(F_masked, axis=(0, 2, 3))
            sum_sq_x += np.nansum(X_masked**2 * mask_broadcast, axis=(0, 2, 3))
            sum_sq_f += np.nansum(F_masked**2 * mask_broadcast, axis=(0, 2, 3))
        
        # Calculate mean
        meanx = sum_x / total_pixels
        meanf = sum_f / total_pixels
        
        # Calculate std using accumulated sum of squares
        stdx = np.sqrt(sum_sq_x / total_pixels - meanx**2)
        stdf = np.sqrt(sum_sq_f / total_pixels - meanf**2)
        
        meanf[3] = 0.0  # icy fraction not normalized
        stdf[3] = 1.0  # icy fraction not normalized
        
        if self.OPTION == 3:
            meanx[-1] = 0 # Set mean of crossing sea criterion to 0 
            stdx[-1] = 1 # Set std of crossing sea criterion to 1 
        
        return meanx, stdx, meanf, stdf
    
    def compute_clim_map(self, chunk_size=32, circular_x=()):
        """Per-pixel mean and std across all files, one chunk at a time.

        Each pixel is averaged only over the steps where it is valid (ice-free
        if use_icymask, else ocean), i.e. divided by its own valid count rather
        than the total number of steps; pixels never valid come out NaN.
        X channels listed in `circular_x` are angles in degrees: their mean is
        the circular mean (atan2 of the summed sin/cos, in [0, 360)) and their
        std is the circular std sqrt(-2 ln R), in degrees."""
        circular_x = list(circular_x)

        shape_x, shape_f = self.X_files[0].shape[1:], self.F_files[0].shape[1:]
        count    = np.zeros((1,) + shape_x[1:], dtype=np.float64)
        sum_x    = np.zeros(shape_x, dtype=np.float64)
        sum_sq_x = np.zeros(shape_x, dtype=np.float64)
        sum_f    = np.zeros(shape_f, dtype=np.float64)
        sum_sq_f = np.zeros(shape_f, dtype=np.float64)
        sum_sin  = np.zeros((len(circular_x),) + shape_x[1:], dtype=np.float64)
        sum_cos  = np.zeros_like(sum_sin)

        for x_file, f_file in tqdm(zip(self.X_files, self.F_files),
                                   total=len(self.X_files), desc='clim map'):
            n = x_file.shape[0]
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                x_chunk = np.array(x_file[start:end], dtype=np.float32)
                f_chunk = np.array(f_file[start:end], dtype=np.float32)

                if self.use_icymask:
                    mask = f_chunk[:, 2:3, :, :].astype(bool)  # N×1×H×W
                else:
                    mask = np.broadcast_to(self.landmask[np.newaxis, np.newaxis].astype(bool),
                                           (end - start, 1) + shape_x[1:])  # N×1×H×W

                # np.where rather than multiplying by the mask, so NaNs on land don't propagate
                x_m = np.where(mask, x_chunk, 0.0)
                f_m = np.where(mask, f_chunk, 0.0)

                count    += mask.sum(0)
                sum_x    += x_m.sum(0)
                sum_sq_x += (x_m * x_m).sum(0)
                sum_f    += f_m.sum(0)
                sum_sq_f += (f_m * f_m).sum(0)
                if circular_x:
                    ang = np.deg2rad(x_chunk[:, circular_x])
                    sum_sin += np.where(mask, np.sin(ang), 0.0).sum(0)
                    sum_cos += np.where(mask, np.cos(ang), 0.0).sum(0)

                del x_chunk, f_chunk, x_m, f_m, mask

        with np.errstate(invalid='ignore', divide='ignore'):
            n_valid = np.where(count > 0, count, np.nan)
            meanx = sum_x / n_valid
            stdx  = np.sqrt(np.maximum(sum_sq_x / n_valid - meanx**2, 0.0))
            meanf = sum_f / n_valid
            stdf  = np.sqrt(np.maximum(sum_sq_f / n_valid - meanf**2, 0.0))
            if circular_x:
                s, c = sum_sin / n_valid, sum_cos / n_valid
                meanx[circular_x] = np.rad2deg(np.arctan2(s, c)) % 360.0
                R = np.clip(np.sqrt(s**2 + c**2), 1e-12, 1.0)
                stdx[circular_x] = np.rad2deg(np.sqrt(-2.0 * np.log(R)))

        return (meanx.astype(np.float32), stdx.astype(np.float32),
                meanf.astype(np.float32), stdf.astype(np.float32))
    
    def invert_x(self, x):
        # x of shape C*H*W (e.g. 4*320*320) normalized
        x = self.inv_tf_x(x)
        if self.OPTION in [1, 2]:
            x[0] = torch.expm1(x[0])   # inverse of log1p
        elif self.OPTION == 3:
            x[0] = torch.expm1(x[0])   # inverse of log1p
            x[3] = torch.expm1(x[3])   # inverse of log1p
        return x
    
    def invert_f(self, f):
        # f of shape C*H*W
        f = self.inv_tf_f(f)
        return f
    
    def transform_x(self, x):
        # x of shape C*H*W (e.g. 4*320*720) in physical unit
        x[0] = torch.log1p(x[0]) 
        x = self.tf_x(x)
        return x
    
    def transform_f(self, f):
        f = self.tf_f(f)
        return f

    def __getitem__(self, idx):
        """Get item by global index - automatically finds correct file."""
        # Map global index to file and local index
        file_idx, local_idx = self._get_file_and_local_idx(idx)
           
        # Load from the appropriate file
        # x = self.X_files[file_idx][local_idx].copy()
        # x[0] = np.log1p(x[0])
        # x = torch.from_numpy(x).float()
        
        if self.OPTION == 1:
            x_raw = self.X_files[file_idx][local_idx][[0,1,2]] # height, period, direction
            x = torch.from_numpy(x_raw).float().clone()
            x[0] = torch.log1p(x[0]) 
            f = torch.from_numpy(self.F_files[file_idx][local_idx][[0,1,3,4]]).float()
            icymask = torch.from_numpy(self.F_files[file_idx][local_idx][[2]]).float()

        if self.OPTION == 2:
            x_raw = self.X_files[file_idx][local_idx][[0,4,5,6,7]]
            x = torch.from_numpy(x_raw).float().clone()
            x[0] = torch.log1p(x[0]) 
            f = torch.from_numpy(self.F_files[file_idx][local_idx][[0,1,3,4]]).float()
            icymask = torch.from_numpy(self.F_files[file_idx][local_idx][[2]]).float()
            
        if self.OPTION == 3:
            x_raw = self.X_files[file_idx][local_idx]
            x = torch.from_numpy(x_raw).float().clone()
            x[0] = torch.log1p(x[0]) 
            x[3] = torch.log1p(x[3])
            f = torch.from_numpy(self.F_files[file_idx][local_idx][[0,1,3,4]]).float()
            icymask = torch.from_numpy(self.F_files[file_idx][local_idx][[2]]).float()
        
        # Apply transforms
        x = self.tf_x(x)
        f = self.tf_f(f)
       
        return x, f, icymask


### Masked dataset functions and class. 

class FillNaN(object):
    """ Replace NaNs with zeros (or any value). """
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self, tensor):
        return torch.nan_to_num(tensor, nan=self.value)

class Mask(object):
    """ Mask land. Let's say it works only on C*H*W. """
    def __init__(self, mask):
        self.mask = mask
    def __call__(self, tensor):
        return tensor * self.mask.to(tensor.device)
        
class npyDataResized(MultiFileNpyData):
    def __init__(self, *args, resize_x=(320,320), resize_f=(320,320), **kwargs):
        super().__init__(*args, **kwargs)  # call original __init__

        self.landmask_original = torch.tensor(self.landmask.astype(bool))[None, :, :].float()
        self.landmask_resized = tf.Resize(resize_x)(self.landmask_original)
        self.original_H = self.X_files[0].shape[2]
        self.original_W = self.X_files[0].shape[3]
        
        # Patch the transforms
        self.tf_x = tf.Compose([
            FillNaN(0.0),
            tf.Resize(resize_x),
            tf.Normalize(self.meanx.tolist(), self.stdx.tolist()),
            Mask(self.landmask_resized)
        ])
        self.tf_f = tf.Compose([
            FillNaN(0.0),
            tf.Resize(resize_f),
            tf.Normalize(self.meanf.tolist(), self.stdf.tolist()),
            Mask(self.landmask_resized)
        ])
        self.tf_icymask = tf.Resize(resize_f)  # for resizing icymask separately if needed
        # Inverse transforms
        self.inv_tf_x = tf.Compose([
            tf.Resize((self.original_H, self.original_W)),
            tf.Normalize(mean=[0]*len(self.meanx), std=(1/self.stdx).tolist()),
            tf.Normalize(mean=(-self.meanx).tolist(), std=[1]*len(self.stdx)),
            Mask(self.landmask_original)
        ])
        self.inv_tf_f = tf.Compose([
            tf.Resize((self.original_H, self.original_W)),
            tf.Normalize(mean=[0]*len(self.meanf), std=(1/self.stdf).tolist()),
            tf.Normalize(mean=(-self.meanf).tolist(), std=[1]*len(self.stdf)),
            Mask(self.landmask_original)
        ])
    
    def __getitem__(self, idx):
        x, f, mask = super().__getitem__(idx)
        mask = self.tf_icymask(mask)
        return x, f, mask


### With wind history.
class npyDataWndHist(npyDataResized):
    """Appends past wind (u, v) snapshots to the forcing tensor as extra channels.

    Historical samples run from `hist_duration_days - hist_freq_days` days
    before the current step up to the current step itself, spaced
    `hist_freq_days` apart, and are appended oldest first, e.g.
    hist_duration_days=5, hist_freq_days=1 -> wind from 4,3,2,1,0 days ago
    (5 samples, daily); hist_duration_days=10, hist_freq_days=2 -> wind from
    8,6,4,2,0 days ago (5 samples, every other day). The last history pair
    therefore repeats the current wind already in channels 0-1. (Checkpoints
    were trained with this layout, so keep it unless retraining.) `padding`
    is still hist_duration_days of steps, so the first usable step is
    hist_duration_days in. Lags are in concatenated-index steps, so they are
    exact only when consecutive files are consecutive months.
    `hist_duration_days` must be a multiple of `hist_freq_days`.
    Source data is 3-hourly (SAMPLES_PER_DAY=8 steps/day).
    """
    SAMPLES_PER_DAY = 8

    def __init__(self, file_list,  # List of tuples: [(Xname, Fname), ...] for each month
        landmaskname=None, use_icymask=True,
        compute_stats=True,
        meanx=None, stdx=None, meanf=None, stdf=None,
        resize_x=None, resize_f=None, OPTION=1,
        hist_duration_days=5, hist_freq_days=1):
        super().__init__(
            file_list, landmaskname, use_icymask,
            compute_stats, meanx, stdx, meanf, stdf, OPTION=OPTION)

        if hist_duration_days % hist_freq_days != 0:
            raise ValueError("hist_duration_days must be a multiple of hist_freq_days")
        self.n_hist = hist_duration_days // hist_freq_days
        hist_step = hist_freq_days * self.SAMPLES_PER_DAY  # index steps between historical samples
        self.hist_offsets = [hist_step * k for k in range(1, self.n_hist + 1)]
        self.padding = self.n_hist * hist_step  # = hist_duration_days * SAMPLES_PER_DAY

        # Expand scaling for wind history (2 channels per historical sample)
        for _ in range(self.n_hist):
            self.meanf = torch.cat([self.meanf, self.meanf[0:2]], dim=0)
        for _ in range(self.n_hist):
            self.stdf = torch.cat([self.stdf, self.stdf[0:2]], dim=0)

        self.tf_x = tf.Compose([
            FillNaN(0.0),
            tf.Resize(resize_x),
            tf.Normalize(self.meanx.tolist(), self.stdx.tolist()),
            Mask(self.landmask_resized)
        ])
        self.tf_f = tf.Compose([
            FillNaN(0.0),
            tf.Resize(resize_f),
            tf.Normalize(self.meanf.tolist(), self.stdf.tolist()),
            Mask(self.landmask_resized)
        ])
        # Inverse transforms
        self.inv_tf_x = tf.Compose([
            tf.Resize((self.original_H, self.original_W)),
            tf.Normalize(mean=[0]*len(self.meanx), std=(1/self.stdx).tolist()),
            tf.Normalize(mean=(-self.meanx).tolist(), std=[1]*len(self.stdx)),
            Mask(self.landmask_original)
        ])
        self.inv_tf_f = tf.Compose([
            tf.Resize((self.original_H, self.original_W)),
            tf.Normalize(mean=[0]*len(self.meanf), std=(1/self.stdf).tolist()),
            tf.Normalize(mean=(-self.meanf).tolist(), std=[1]*len(self.stdf)),
            Mask(self.landmask_original)
        ])

    def __len__(self):
        return self.total_length - self.padding

    def __getitem__(self, idx):
        """Get item by global index - automatically finds correct file."""
        # Map global index to file and local index
        file_idx, local_idx = self._get_file_and_local_idx(idx + self.padding)

        if self.OPTION == 1:
            x_raw = self.X_files[file_idx][local_idx][[0,1,2]] # height, period, direction
            x = torch.from_numpy(x_raw).float().clone()
            x[0] = torch.log1p(x[0])
            f = torch.from_numpy(self.F_files[file_idx][local_idx][[0,1,3,4]]).float()
            icymask = torch.from_numpy(self.F_files[file_idx][local_idx][[2]]).float()

        if self.OPTION == 2:
            x_raw = self.X_files[file_idx][local_idx][[0,4,5,6,7]]
            x = torch.from_numpy(x_raw).float().clone()
            x[0] = torch.log1p(x[0])
            f = torch.from_numpy(self.F_files[file_idx][local_idx][[0,1,3,4]]).float()
            icymask = torch.from_numpy(self.F_files[file_idx][local_idx][[2]]).float()

        if self.OPTION == 3:
            x_raw = self.X_files[file_idx][local_idx]
            x = torch.from_numpy(x_raw).float().clone()
            x[0] = torch.log1p(x[0])
            x[3] = torch.log1p(x[3])
            f = torch.from_numpy(self.F_files[file_idx][local_idx][[0,1,3,4]]).float()
            icymask = torch.from_numpy(self.F_files[file_idx][local_idx][[2]]).float()

        for offset in self.hist_offsets:
            file_idx_hist, local_idx_hist = self._get_file_and_local_idx(idx + offset)
            f_hist = torch.from_numpy(self.F_files[file_idx_hist][local_idx_hist]).float()
            f = torch.cat([f, f_hist[[0,1], :, :]], dim=0)  # Append historical wind speed and direction

        # Apply transforms
        x = self.tf_x(x)
        f = self.tf_f(f)
        icymask = self.tf_icymask(icymask)

        return x, f, icymask

    

