import argparse
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        prog='''
        Samples and removes extra columns from crowns geopackage.
        example:
        python make_tile_sample.py \\
            --crowns=huc180102111105/crowns/10TDL0464245187.gpkg \\
            --outfile=huc180102111105/10TDL0464245187_labels.gpkg \\
            -c 'IDdalponte' 'zmax' \\
            --label_col='label' \\
            
        '''
        )
    
    parser.add_argument(
        '--crowns',
        type=str,
        required=True,
        help='Path to crowns that will be sampled'
        )
    
    parser.add_argument(
        '--outfile',
        type=str,
        required=True,
        help='path where sample will be written.'
        ) 
    
    parser.add_argument(
        '-c',
        '--include_cols',
        type=str,
        nargs='+',
        required=True,
        help='Names of columns that will be kept in label gpkg.'
        )
    
    parser.add_argument(
        '--label_col',
        type=str,
        required=True,
        help='Name of column that will be added for labels.'
        )
    
    parser.add_argument(
        '--frac',
        type=float,
        required=False,
        default=0.5,
        help='fraction of crowns to be included in sample. Default=0.5 .'
        )
    
    return parser.parse_args()  


def sample_etc(crowns, outfile, include_cols, label_col, frac):
    df = gpd.read_file(crowns)[include_cols + ['geometry']]
    df[label_col] = np.nan
    df = df.sample(frac=frac)
    df.to_file(outfile)
    

if __name__ == '__main__':
    
    args = parse_args()
    
   
    sample_etc(
        args.crowns,
        args.outfile,
        args.include_cols,
        args.label_col,
        args.frac
        )