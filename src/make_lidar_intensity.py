#!/bin/bash/python
#
# requires pdal-python
# conda install -c conda-forge python-pdal
# Usage:
#    
# cat tmp/tiles.list | parallel --progress \
#   python src/make_lidar_intensity.py  \
#   --tile={} \
#   --intensity_dir=intensity \
#   --dem_dir=DEM  \
#   --chm_dir=CHM \
#   --crs='EPSG:32610' \
#   --hag_limit=71

from pathlib import Path
import argparse
import pdal


def parse_arguments():
    '''parses the arguments, returns args'''

    # init parser
    parser = argparse.ArgumentParser()

    # add args
    parser.add_argument(
        '--tile',
        type=str,
        required=True,
        help='Path to Lidar tile.'
    )

    parser.add_argument(
        '--intensity_dir',
        type=str,
        required=True,
        help='Path to intensity output directory.'
    )
    
    parser.add_argument(
        '--chm_dir',
        type=str,
        required=True,
        help='Path to CHM output directory.'
    )
        
    parser.add_argument(
        '--dem_dir',
        type=str,
        required=True,
        help='Path to DEM output directory.'
    )
    
    parser.add_argument(
        '--resolution',
        type=float,
        required=False,
        default=3.0,
        help='Resolution of output tif. Default=3.0, to match Planetscope.'
    )
    
    parser.add_argument(
        '--hag_limit',
        type=float,
        required=False,
        default=120,
        help='HaightAboveGround threshold above which points will be considered noise.'
    )
    
    parser.add_argument(
        '--crs',
        type=str,
        required=False,
        default=None,
        help='Coordinate reference system for output if reprojection is desired.'
    )
    
    
    
    # parse the args
    args = parser.parse_args()

    return(args)


def intense_pipe(
    tile,
    intensity_dir,
    chm_dir,
    dem_dir,
    resolution,
    hag_limit,
    crs=None
    ):
    '''
    writes 1 m tif from mean intensity of first returns.
    '''
    
    # define stages
    reader = pdal.Reader.las(
        tile)
    
    noise_filter = pdal.Filter.range(
        limits='Classification![7:7],Classification![18:18]',
    )
    
    reproject = pdal.Filter.reprojection(
        out_srs=crs
    )
    
    hag = pdal.Filter.hag_nn(count=2)
    
    chm_writer = pdal.Writer.gdal(
        filename= str(Path(chm_dir) / f'{Path(tile).stem}.tif'),
        data_type='float',
        dimension='HeightAboveGround',
        where=f'HeightAboveGround < {hag_limit}',
        output_type='max',
        resolution=str(resolution)
        )
    
    intensity_writer = pdal.Writer.gdal(
        filename= str(Path(intensity_dir) / f'{Path(tile).stem}.tif'),
        dimension='Intensity',
        data_type='uint16_t',
        output_type='mean',
        where=f'(ReturnNumber == 1) && (HeightAboveGround < {hag_limit})',
        resolution=str(resolution)
        )
    
    dem_writer = pdal.Writer.gdal(
        filename= str(Path(dem_dir) / f'{Path(tile).stem}.tif'),
        data_type='uint16_t',
        output_type='mean',
        where='Classification == 2',
        resolution=str(resolution)
        )
        
    # create pipeline from stages
    pipeline = pdal.Pipeline()
    pipeline |= reader
    pipeline |= noise_filter
    if crs is not None:
        pipeline |= reproject
    pipeline |= hag
    pipeline |= chm_writer
    pipeline |= intensity_writer
    pipeline |= dem_writer
    
    # execute pipeline
    pipeline.execute()
    

if __name__ == '__main__':
    args = parse_arguments()
    
    intense_pipe(
        args.tile,
        args.intensity_dir,
        args.chm_dir,
        args.dem_dir,
        args.resolution,
        args.hag_limit,
        crs=args.crs
    )