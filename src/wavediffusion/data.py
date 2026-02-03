import torch
import csv
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as tf

## Dataset utility functions

# Mainly used to discard labels and only output data
class MappedDataset(Dataset):
    def __init__(self, dataset, fn):
        self.dataset = dataset
        self.fn = fn
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        return self.fn(self.dataset[i])

img_train_transform = tf.Compose([
    tf.RandomHorizontalFlip(),
    tf.ToTensor(),
    tf.Lambda(lambda t: (t * 2) - 1)
])

img_normalize = lambda x: ((x + 1)/2).clamp(0, 1)

# My custom datasets
# Note: should be multiworker friendly. Only numpy or CPU tensor.
# ice free for now
class npyData(Dataset):
    def __init__(self, 
        Xname, Fname, maskname, 
        llim=0, rlim=-1, # chunking if needed
        compute_stats=True, # if to compute mean/std from data
        meanx=None, stdx=None, meanf=None, stdf=None # if passing precomputed stats (as numpy arrays of shape (C,))
    ):
        super().__init__()
        # Load X and F using memory mapping
        self.X = np.load(Xname, mmap_mode="r")[llim:rlim]
        self.F = np.load(Fname, mmap_mode="r")[llim:rlim, 0:3] 
        # Load mask directly
        self.mask = np.load(maskname)
        # Mean and std for normalization
        if compute_stats:
            print("Computing dataset mean and std...")
            meanx, stdx, meanf, stdf = self._compute_mean_std()
        else:
            assert meanx is not None and stdx is not None, "Must provide meanx/stdx when compute_stats=False"
            assert meanf is not None and stdf is not None, "Must provide meanf/stdf when compute_stats=False"
        # Store as tensors (can be directly passed in)
        self.meanx = torch.as_tensor(meanx, dtype=torch.float32)
        self.stdx = torch.as_tensor(stdx, dtype=torch.float32)
        self.meanf = torch.as_tensor(meanf, dtype=torch.float32)
        self.stdf = torch.as_tensor(stdf, dtype=torch.float32)
        # Final transform including normalization, can be different for X and F
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

    def __len__(self):
        return self.X.shape[0] - 1 # to prevent the last sample? Because Y = X[idx+1]
    
    def _compute_mean_std(self):
        # Compute channel-wise mean and std for X and F over masked pixels only.
        # Assumes X, F have shape (N, C, H, W) and mask has shape (H, W)
        mask = self.mask.astype(bool)  # ensure boolean
        mask_broadcast = mask[None, None, :, :]  # shape (1,1,H,W) to broadcast over N,C
        # Apply mask
        X_masked = self.X * mask_broadcast
        F_masked = self.F * mask_broadcast
        # Count of valid pixels per channel
        n_pixels = mask.sum() * self.X.shape[0]
        # channel-wise mean
        meanx = np.nansum(X_masked, axis=(0,2,3)) / n_pixels
        meanf = np.nansum(F_masked, axis=(0,2,3)) / n_pixels
        # channel-wise std
        stdx = np.sqrt(np.nansum(((X_masked - meanx[None,:,None,None])**2) * mask_broadcast, axis=(0,2,3)) / n_pixels)
        stdf = np.sqrt(np.nansum(((F_masked - meanf[None,:,None,None])**2) * mask_broadcast, axis=(0,2,3)) / n_pixels)
        return meanx, stdx, meanf, stdf

    def __getitem__(self, idx):
        # Convert np -> tensor first
        x = torch.from_numpy(self.X[idx]).float()
        f = torch.from_numpy(self.F[idx]).float()
        # Apply transforms
        x = self.tf_x(x)
        f = self.tf_f(f)
        return x, f


## Toy datasets

class Swissroll(Dataset):
    def __init__(self, tmin, tmax, N, center=(0,0), scale=1.0):
        t = tmin + torch.linspace(0, 1, N) * tmax
        center = torch.tensor(center).unsqueeze(0)
        self.vals = center + scale * torch.stack([t*torch.cos(t)/tmax, t*torch.sin(t)/tmax]).T

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, i):
        return self.vals[i]

class DatasaurusDozen(Dataset):
    def __init__(self, csv_file, dataset, enlarge_factor=15, delimiter='\t', scale=50, offset=50):
        self.enlarge_factor = enlarge_factor
        self.points = []
        with open(csv_file, newline='') as f:
            for name, *rest in csv.reader(f, delimiter=delimiter):
                if name == dataset:
                    point = torch.tensor(list(map(float, rest)))
                    self.points.append((point - offset) / scale)

    def __len__(self):
        return len(self.points) * self.enlarge_factor

    def __getitem__(self, i):
        return self.points[i % len(self.points)]

def interpolate_polyline(points, num_samples):
    """
    Given a list of 2D points defining a polyline,
    sample num_samples points uniformly along its arc length.
    """
    points = np.array(points)
    # Compute distances between consecutive points
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumdist = np.concatenate(([0], np.cumsum(dists)))
    total_length = cumdist[-1]
    # Equally spaced arc-length values
    sample_dists = np.linspace(0, total_length, num_samples)
    samples = []
    for d in sample_dists:
        # Find which segment d falls in
        seg = np.searchsorted(cumdist, d, side='right') - 1
        seg = min(seg, len(dists) - 1)
        # Compute local interpolation parameter
        t = (d - cumdist[seg]) / dists[seg] if dists[seg] > 0 else 0
        sample = (1 - t) * points[seg] + t * points[seg + 1]
        samples.append(sample)
    return np.array(samples)

class TreeDataset(Dataset):
    def __init__(self, branching_factor=4, depth=3, num_samples_per_path=30):
        """
        Initializes a tree dataset where each leaf of the tree lies on the
        circle of radius 1. The tree is constructed with the given branching_factor
        and depth. Each leaf’s path is sampled uniformly, and each sampled point
        is given the label of the leaf.

        Parameters:
         - branching_factor (int): number of branches at each node.
         - depth (int): number of branchings (excluding the root).
                        Total leaves = branching_factor ** depth.
         - num_samples_per_path (int): number of points sampled along each path.
        """
        self.data = []
        self.total_leaves = branching_factor ** depth

        # Iterate over each leaf index
        for i in range(self.total_leaves):
            # Build the sequence of nodes along the path from the root to this leaf.
            # Start with the root at (0, 0)
            path_points = [np.array([0.0, 0.0])]

            # For each level l (1 to depth), compute the branch node.
            for l in range(1, depth + 1):
                # Group size for this level (leaves per branch node)
                group_size = branching_factor ** (depth - l)  # For l == depth, group_size == 1.
                # A_l is the branch index for level l
                A_l = i // group_size
                # Compute the average index for all leaves under this branch node
                avg_index = A_l * group_size + (group_size - 1) / 2.0
                # Compute angular coordinate (all leaves are uniformly spaced on the circle)
                theta = avg_index * (2 * np.pi / self.total_leaves)
                # Set radius proportional to the level (leaf at level==depth has r==1)
                r = l / depth
                p = np.array([r * np.cos(theta), r * np.sin(theta)])
                path_points.append(p)

            # Sample points uniformly along the polyline defined by the path
            samples = interpolate_polyline(path_points, num_samples_per_path)
            # Append each sample with its label (the leaf index)
            for sample in samples:
                # Each item is a tuple: (2D coordinate tensor, label)
                self.data.append((torch.tensor(sample, dtype=torch.float32), i))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
