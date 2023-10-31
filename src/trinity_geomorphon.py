#!/bin/python

import whitebox
wbt = whitebox.WhiteboxTools()

import argparse
import os
import temppathlib

import geopandas as gpd
from tqdm import tqdm
import rioxarray
from rioxarray.exceptions  import NoDataInBounds


def parse_args():
    parser = argparse.ArgumentParser(
        prog='''
        Makes geomorphons for each sub-basin found in vector found at
        huc10_path (+ 100m buffer).  Sub-Basin geomorphons can later
        be put together as a vrt.
        
        Uses 100 m, 250 m, 500 m, 1000 m,
        and 2000 m outer r.  Inner r are 1/2 of outer r. 
        Flatness threshold of 0 is used.
        
        example:
        python make_tile_sample.py \\
            ---dem_path=/home/michael/tmp/trinity_basin_carrhirzdelta_dem.tif \\
            --huc10_path=Path('/media/storage/watershed_boundaries/trinity_river_huc10.geojson') \\
            --out_dir=/home/michael/TreeMortality/geomorphons
        '''
        )
    
    
    parser.add_argument(
        '--dem_path',
        type=str,
        required=True,
        help='Path to DEM'
        )
    
    parser.add_argument(
        '--huc10_path',
        type=str,
        required=True,
        help='Path to sub-basins vector'
        )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='Path to output directory'
        )

    return parser.parse_args()  


def my_callback(value):
    #if not '%' in value:
    #print(value)
    pass


def geomorph_all_scales(dem_path, out_dir, basin):
    for r in tqdm([100, 250, 500, 1000, 2000]):

        output = out_dir / f'geomorph_{basin}_{r}.tif' 
        
        wbt.geomorphons(
            dem_path, 
            output, 
            search=r, 
            threshold=0.0, 
            fdist=0, 
            skip=round(r / 2), 
            forms=True, 
            residuals=False, 
            callback=my_callback
        )
        
        
if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # read huc10s to df
    huc10s = gpd.read_file(args.huc10_path)[['huc10', 'geometry']]
    
    with temppathlib.TemporaryDirectory() as tmp_dir:
        for i, row in huc10s.iterrows():
            print(f'on row {i + 1} of {len(huc10s)}')   
            # get bounds of basin
            minx, miny, maxx, maxy = row.geometry.bounds
            # read DEM clipped to bounds
            try:
                dem = rioxarray.open_rasterio(args.dem_path).rio.clip_box(
                    minx=minx,
                    miny=miny,
                    maxx=maxx,
                    maxy=maxy
                )
            except NoDataInBounds:
                print(f'{row.huc10} ({row.huc10}) failed with NoDataInBounds')
                continue
                
            # clip to 100m buffered huc10 basin
            dem = dem.rio.clip([row.geometry.buffer(100)])
            # write a tmp dem
            tmp_dst = tmp_dir.path / f'{row.huc10}.tif'
            dem.rio.to_raster(tmp_dst)
            # delete dem to free memory
            del dem
            # make geomorphons of all scales
            geomorph_all_scales(tmp_dst, args.out_dir, row.huc10)