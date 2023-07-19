#%%
import geopandas as gpd
import pandas as pd
import xarray as xr
import rioxarray
import numpy as np
import json

# %%
crs = 'EPSG:26910'

helena_cc_path ='/media/storage/RAVG/helena/ca4077112312420170831_20151022_20171027_rdnbr_cc.tif'
helena = rioxarray.open_rasterio(helena_cc_path).rio.reproject(crs)

monument_cc_path = '/media/storage/RAVG/TrinityCounty/2021/ca4075212333720210731/ca4075212333720210731_20201014_20211009_rdnbr_cc.tif'
monument = rioxarray.open_rasterio(monument_cc_path).rio.reproject(crs)

aoi = '/home/michael/TreeMortality/data/helena/helena.gpkg'
aoi = gpd.read_file(aoi)
minx, miny, maxx, maxy = aoi.total_bounds

helena_raster = helena.rio.clip_box(
    minx=minx,
    miny=miny,
    maxx=maxx,
    maxy=maxy,
)

helena_raster = helena_raster.where(
    helena_raster[0] != helena_raster.attrs['_FillValue']
    )

monument_raster = monument.rio.clip_box(
    minx=minx,
    miny=miny,
    maxx=maxx,
    maxy=maxy,
).rio.reproject_match(helena_raster)

monument_raster = monument_raster.where(
    monument_raster[0] != monument_raster.attrs['_FillValue']
    )
#%%
codes = {
    # exclude 0 here
    1: (1/3, 0),
    2: (2/3, 0),
    3: (1, 0),
    4: (0, 1/3),
    5: (1/3, 1/3),
    6: (2/3, 1/3),
    7: (1, 1/3),
    8: (0, 2/3),
    9: (1/3, 2/3),
    10: (2/3, 2/3),
    11: (1, 2/3),
    12: (0, 1),
    13: (1/3, 1),
    14: (2/3, 1),
    15: (1, 1)
}


treatments = xr.Dataset(
    {
        'monument': monument_raster[0],
        'helena': helena_raster[0]
    }
)

treatments['code'] = (('y', 'x'), np.zeros_like( monument_raster[0]))

for key in codes.keys():
    treatments['code'] = treatments['code'].where(
        ~(treatments.code == 0)
        & ~(treatments.helena <= codes[key][0])
        & ~(treatments.monument <= codes[key][1]),
        other=key
    )
    

# %%
