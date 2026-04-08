import numpy as np
import xarray as xr

# Open a random raw file to get lat and lon
ds = xr.open_dataset('/global/homes/j/jiarongw/scratch_folder/wave_data/raw/LOPS_WW3-GLOB-30M_202109.nc')
lat_deg = ds.latitude.values[1:-2]
lon_deg = ds.longitude.values

def assemble (hs, lp, theta, wnd, ice):

    global lat_deg, lon_deg    
    deg2rad = np.pi / 180.0
    R = 6371e3

    # Unit vector
    ux = -np.sin(np.deg2rad(theta))
    uy =  np.cos(np.deg2rad(theta))
    
    wave = xr.Dataset({
        'hs': (('lat', 'lon'), hs),
        'lp': (('lat', 'lon'), lp),
        'theta': (('lat', 'lon'), theta),
        'wnd': (('lat', 'lon'), wnd),
        'ux': (('lat', 'lon'), ux),
        'uy': (('lat', 'lon'), uy),
        'ice': (('lat', 'lon'), ice)
    }, coords={'lat': lat_deg, 'lon': lon_deg})

    # xr.differentiate gives d(hs)/d(lat_deg), so divide by R * deg2rad factor
    # ∂hs/∂y = (1/R) * (1/deg2rad) * ∂hs/∂lat_deg  →  divide by R·(π/180)
    # ∂hs/∂x = (1/(R·cos(lat_rad))) * (1/deg2rad) * ∂hs/∂lon_deg
    
    dhs_dlat = wave.hs.differentiate('lat')   # d(hs)/d(lat_deg)
    dhs_dlon = wave.hs.differentiate('lon')   # d(hs)/d(lon_deg)
    
    dhs_dy = dhs_dlat / (R * deg2rad)
    # dhs_dx = dhs_dlon / (R * deg2rad * np.cos(np.deg2rad(wave.lat.values)[:, np.newaxis]))
    dhs_dx = dhs_dlon / (R * deg2rad * np.cos(np.deg2rad(wave.lat)))
        
    # Project onto theta direction
    wave['grad'] = dhs_dx * ux + dhs_dy * uy
    wave['slope'] = np.pi**2 * wave.hs / 9.8 / (wave.lp/1.56)
    wave['hs_hat'] = 9.8 * wave.hs / wave.wnd ** 2
    wave['tp_hat'] = 9.8 * (wave.lp/1.56) ** 0.5 / (2 * np.pi * wave.wnd)

    return wave

