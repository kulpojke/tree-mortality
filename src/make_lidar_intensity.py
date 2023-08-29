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
# cat tmp/tiles.list | parallel --progress \
    # python src/make_lidar_intensity.py \
    # --tile={} \
    # --intensity_dir=intensity \
    # --ground_dir=ground_map \
    # --hag_threshold=100


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
        '--groundiness_dir',
        type=str,
        required=True,
        help='Path to groundiness map output directory.'
    )
    
    parser.add_argument(
        '--resolution',
        type=float,
        required=False,
        default=3.0,
        help='Resolution of output tif. Default=3.0, to match Planetscope.'
    )
    
    parser.add_argument(
        '--hag_threshold',
        type=float,
        required=False,
        default=120,
        help='HaightAboveGround threshold above which points will be considered noise.'
    )
    
    # parse the args
    args = parser.parse_args()

    return(args)


def intense_pipe(
    tile,
    intensity_dir,
    chm_dir,
    dem_dir,
    groundiness_dir,
    resolution
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
    
    hag = pdal.Filter.hag_nn(count=2)
    
    chm = pdal.Writer.gdal(
        filename= str(Path(chm_dir) / f'{Path(tile).stem}.tif'),
        data_type='float',
        dimension='HeightAboveGround',
        output_type='max',
        resolution=str(resolution)
        )
    
    intensity = pdal.Writer.gdal(
        filename= str(Path(intensity_dir) / f'{Path(tile).stem}.tif'),
        dimension='Intensity',
        data_type='uint16_t',
        output_type='mean',
        where='ReturnNumber == 1',
        resolution=str(resolution)
        )
    
    dem_writer = pdal.Writer.gdal(
        filename= str(Path(dem_dir) / f'{Path(tile).stem}.tif'),
        data_type='uint16_t',
        output_type='mean',
        where='Classification == 2',
        resolution=str(resolution)
        )
    
    
    
    
    
    ground = pdal.Filter.range(limits='Classification[2:2]')
    
    ground_writer = pdal.Writer.gdal(
        filename= str(Path(groundiness_dir) / f'{Path(tile).stem}.tif'),
        data_type='uint16_t',
        output_type='count',
        resolution=str(resolution)
        )
        
    # create pipeline from stages
    pipeline = pdal.Pipeline()
    pipeline |= reader
    pipeline |= noise_filter
    pipeline |= hag
    pipeline |= chm
    pipeline |= intensity
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
        args.groundiness_dir,
        args.resolution
    )