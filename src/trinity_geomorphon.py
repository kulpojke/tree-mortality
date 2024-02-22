#!/bin/python

import whitebox
wbt = whitebox.WhiteboxTools()

import argparse
import os
from pathlib import Path
import temppathlib
from datetime import datetime

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
        python trinity_geomorphon.py \\
            --dem_path=/home/michael/tmp/trinity_basin_carrhirzdelta_dem.tif \\
            --huc10_path=/media/storage/watershed_boundaries/trinity_river_huc10.geojson \\
            --out_dir=/home/michael/TreeMortality/trinity/geomorphons
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

    args =  parser.parse_args()  
    args.dem_path = Path(args.dem_path)
    args.huc10_path = Path(args.huc10_path)
    args.out_dir = Path(args.out_dir)
    
    return args

    
def my_callback(value):
    #if not '%' in value:
    #print(value)
    pass


def geomorph_all_scales(dem_path, out_dir, basin):
    for r in tqdm([100, 250, 500, 1000, 2000]):

        output = str(out_dir / f'geomorph_{basin}_{r}.tif' )
        
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
    
    with open(args.out_dir / 'geomorph.log', 'a') as log_dst:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_dst.write(f'trinity_geomorph.py, run at {now}.\n')
        with temppathlib.TemporaryDirectory() as tmp_dir:
            for i, row in huc10s.iterrows():
                log_dst.write(f'on row {i + 1} ({row.huc10}) of {len(huc10s)}\n')
                # check to make sure huc10 has not already been done
                is_it_file = args.out_dir / f'geomorph_{row.huc10}_2000.tif'
                if is_it_file.is_file():
                    log_dst.write(f'Geomorphons for {row.huc10} have already been calculated.\n')
                    log_dst.write(f'Skipping.\n')
                    continue
                # get bounds of basin
                minx, miny, maxx, maxy = row.geometry.bounds
                # read DEM clipped to bounds
                try:
                    log_dst.write('Reading DEM...\n')
                    dem = rioxarray.open_rasterio(args.dem_path).rio.clip_box(
                        minx=minx,
                        miny=miny,
                        maxx=maxx,
                        maxy=maxy
                    )
                    log_dst.write('read.\n')
                except NoDataInBounds:
                    log_dst.write(f'{row.huc10} ({row.huc10}) failed with NoDataInBounds!!!!\n')
                    continue
                    
                # clip to 100m buffered huc10 basin
                log_dst.write('Clipping DEM...\n')
                dem = dem.rio.clip([row.geometry.buffer(100)])
                log_dst.write('clipped.\n')
                # write a tmp dem
                log_dst.write('Writing tempfile...\n')
                tmp_dst = tmp_dir.path / f'{row.huc10}_{i}.tif'
                dem.rio.to_raster(tmp_dst)
                log_dst.write('written.\n')
                # delete dem to free memory
                log_dst.write('Deleting DEM...\n/n')
                del dem
                log_dst.write('deleted.\n')
                # make geomorphons of all scales
                log_dst.write(f'Making geomorphons for {row.huc10}...\n')
                geomorph_all_scales(tmp_dst, args.out_dir, row.huc10)
                log_dst.write('made.\n\n')