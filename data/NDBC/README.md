## Command for buoy analysis:

* Station 46042 (Monterey Bay) BUOY_LAT = 36.787 BUOY_LON = -122.408 YEAR = 2004, 2022
* Station 51028 (Christmas Island) BUOY_LAT = 0.000 BUOY_LON = -153.913 YEAR = 2004


## Read buoy's moment measurement:

`module load pytorch/2.6.0`

`python read_ndbc_directional.py --station 46042 --year 2004 --input-dir ~/scratch_folder/wave_data/NDBC/`

`mv ndbc_directional_46042_2004.nc ~/scratch_folder/wave_data/NDBC/` 

## Extract from WW3's raw data (optional because diffusion model output directory also has WW3 target)

`python extract_ww3_at_buoy.py --station 46042 --year 2004 --lat 36.787 --lon -122.408`

## Reconstruct and partition the NDBC data

`python buoy_reconstruct_partition.py --input-dir ~/scratch_folder/wave_data/NDBC/ --station 46042 --year 2022`

## Read from diffusion model output data 
Here I used 36 and -123 instead of 37 and -122 because otherwise it seems to be too close to the land and producing weird values due to resizing of the network.

`python gather_diffusion_at_buoy.py /global/homes/j/jiarongw/scratch_folder/final/temp/OPTION3_sigma100_epoch10_2022/ --lat 36 --lon -123`