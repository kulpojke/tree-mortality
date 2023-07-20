#%%
from pathlib import Path
import os
import affine
import rasterio.features
import shapely.geometry as sg
import geopandas as gpd
import pandas as pd
import xarray as xr
import rioxarray
import numpy as np
import json
from dea_tools.spatial import xr_vectorize

# %%
# args
crs = 'EPSG:26910'

#helena_cc_path ='/media/storage/RAVG/helena/ca4077112312420170831_20151022_20171027_rdnbr_cc.tif'
helena_cc_path = '../data/tmp/ca4077112312420170831_20151022_20171027_rdnbr_cc.tif'
helena = rioxarray.open_rasterio(helena_cc_path).rio.reproject(crs)

#monument_cc_path = '/media/storage/RAVG/TrinityCounty/2021/ca4075212333720210731/ca4075212333720210731_20201014_20211009_rdnbr_cc.tif'
monument_cc_path = '../data/tmp/ca4075212333720210731_20201014_20211009_rdnbr_cc.tif'
monument = rioxarray.open_rasterio(monument_cc_path).rio.reproject(crs)

#aoi_path = '/home/michael/TreeMortality/data/helena/helena.gpkg'
aoi_path = '../data/tmp/aoi.gpkg'

out_dir = Path('/home/michael/tmp/treatment_polys')
os.makedirs(out_dir, exist_ok=True)
log_path = out_dir / 'treatment_desc.csv'

#%%
# get bounds from aoi
aoi = gpd.read_file(aoi_path)
minx, miny, maxx, maxy = aoi.total_bounds

# clip rasters to bounds
helena_raster = helena.rio.clip_box(
    minx=minx,
    miny=miny,
    maxx=maxx,
    maxy=maxy,
)

# change no data values to nans
helena_raster = helena_raster.where(
    helena_raster[0] != helena_raster.attrs['_FillValue'],
    ).fillna(0)
helena_raster.rio.to_raster('/home/michael/tmp/HEL.tif')

monument_raster = monument.rio.clip_box(
    minx=minx,
    miny=miny,
    maxx=maxx,
    maxy=maxy,
).rio.reproject_match(
    helena_raster
)

monument_raster = monument_raster.where(
    monument_raster[0] != monument_raster.attrs['_FillValue'],
    ).fillna(0)

monument_raster.rio.to_raster('/home/michael/tmp/MON.tif')
#%%

def severity(fire):
    '''returns fire severity from fraction'''
    if fire == 0:
        return 'unburned'
    if fire <= 1/3:
        return 'low'
    if 1/3 < fire <= 2/3:
        return 'medium'
    if 2/3 < fire:
        return 'high'
    
@np.vectorize
def encode(hel, mon):
    '''
    Code |  Helena  | Monument
    0    | unburned | unburned
    1    | unburned | low
    2    | unburned | medium
    3    | unburned | high
    4    | low      | unburned
    5    | low      | low
    6    | low      | medium
    7    | low      | high
    8    | medium   | unburned
    9    | medium   | low
    10   | medium   | medium
    11   | medium   | high
    12   | high     | unburned
    13   | high     | low
    14   | high     | medium
    15   | high     | high
    '''

    codes = {
        'unburned':{
            'unburned': 0,
            'low': 1,
            'medium': 2,
            'high': 3 
        },
        'low': {
            'unburned': 4,
            'low': 5,
            'medium': 6,
            'high': 7
        },
        'medium': {
            'unburned': 8,
            'low': 9,
            'medium': 10,
            'high': 11
        },
        'high': {
            'unburned': 12,
            'low': 13,
            'medium': 14,
            'high': 15
        }
    }

    return codes[severity(hel)][severity(mon)]
    



# make dataset
treatments = xr.Dataset(
    {
        'monument': monument_raster[0],
        'helena': helena_raster[0]
    }
)
#%%

# add code raster full of zeros
code = encode(helena_raster[0], monument_raster[0])

#
treatments['code'] = (('y', 'x'), code.astype(np.int8))
treatments = treatments.rio.clip_box(
    minx=minx,
    miny=miny,
    maxx=maxx,
    maxy=maxy,
)
treatments.code.rio.to_raster('/home/michael/tmp/CODES.tif')

# plygonize
df = xr_vectorize(da=treatments.code, crs=crs)
df['attribute'] = df.attribute.astype(int)
df['area_'] = df.area
# %%
def make_study_polys(codes, n, out_dir, log_path=None):
    log_dfs = []
    total_area = df.area.sum()
    for code in codes:
        sub = df[
            df.attribute == code
            ].nlargest(n, 'area_')
        
        fname = out_dir / f'code{code}_n{n}.gpkg'

        sub.to_file(fname)
        
        if log_path is not None:
            desc = sub.describe().area_.to_frame().T
            desc['%A'] = 100 * df[df.attribute == code].area.sum() / total_area
            log_dfs.append(desc)

    if log_path is not None:
        log_df = pd.concat(log_dfs)
        log_df.to_csv(log_path)
# %%

codes = range(16)
n = 5
make_study_polys(codes, n, out_dir, log_path=log_path)

# %%
