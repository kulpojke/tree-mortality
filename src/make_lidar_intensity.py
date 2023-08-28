#!/bin/bash/python
#
# requires pdal-python
# conda install -c conda-forge python-pdal
# Usage:
# python make_lidar_intensity.py \
#   --tile=10TEL0509245547.laz \
#   --intensity_dir=/path/to/intensity_dir
#
# Or:
# cat tiles.list | parallel --progress \
#   python make_lidar_intensity.py \
#       --tile={} \
#       --intensity_dir=/path/to/intensity_dir

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
        help='Path to output directory.'
    )
    
    parser.add_argument(
        '--dsm_dir',
        type=str,
        required=True,
        help='Path to output directory.'
    )
    
    parser.add_argument(
        '--resolution',
        type=float,
        required=False,
        default=3.0,
        help='Resolution of output tif. Default=3.0, to match Planetscope.'
    )
    
    # parse the args
    args = parser.parse_args()

    return(args)


def intense_pipe(tile, intensity_dir, dsm_dir, resolution):
    '''
    writes 1 m tif from mean intensity of first returns.
    '''
    
    # define stages
    reader = pdal.Reader.las(tile)
    filter = pdal.Filter.range(
        limits='ReturnNumber[1:1],Classification![7:7],Classification![18:18]')
    intensity = pdal.Writer.gdal(
        filename= str(Path(intensity_dir) / f'{Path(tile).stem}.tif'),
        dimension='Intensity',
        data_type='uint16_t',
        output_type='mean',
        resolution=str(resolution)
        )
    outlier = filter.outlier(
        method='radius',
        radius='1.0',
        min_k='4'
    )
    dsm = pdal.Writer.gdal(
        filename= str(Path(dsm_dir) / f'{Path(tile).stem}.tif'),
        dimension='Z',
        data_type='uint16_t',
        output_type='max',
        resolution=str(resolution)
        )
        
    # create pipeline from stages
    pipeline = pdal.Pipeline()
    pipeline |= reader
    pipeline |= filter
    pipeline |= intensity
    pipeline |= outlier
    pipeline |= dsm

    # execute pipeline
    pipeline.execute()
    

if __name__ == '__main__':
    args = parse_arguments()
    intense_pipe(args.tile, args.intensity_dir, args.dsm_dir, args.resolution)
    