#!/bin/bash/python
#
# requires pdal-python
# conda install -c conda-forge python-pdal
# Usage:
# python make_lidar_intensity.py \
#   --tile=10TEL0509245547.laz \
#   --out_dir=/path/to/out_dir
#
# Or:
# cat tiles.list | parallel --progress \
#   python make_lidar_intensity.py \
#       --tile={} \
#       --out_dir=/path/to/out_dir

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
        '--out_dir',
        type=str,
        required=True,
        help='Path to output directory.'
    )
    
    # parse the args
    args = parser.parse_args()

    return(args)


def intense_pipe(tile, out_dir):
    '''
    writes 1m tif from mean intensity of first returns.
    '''
    
    # define stages
    reader = pdal.Reader.las(tile)
    filter = pdal.Filter.range(limits='ReturnNumber[1:1]')
    writer = pdal.Writer.gdal(
        filename= str(Path(out_dir) / f'{Path(tile).stem}.tif'),
        dimension='Intensity',
        data_type='uint16_t',
        output_type='mean',
        resolution='1'
        )
    
    # create pipeline from stages
    pipeline = pdal.Pipeline()
    pipeline |= reader
    pipeline |= filter
    pipeline |= writer

    # execute pipeline
    pipeline.execute()
    

if __name__ == '__main__':
    args = parse_arguments()
    intense_pipe(args.tile, args.out_dir)
    