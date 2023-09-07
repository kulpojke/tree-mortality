#tif_path=data/helena/NAIP/
#crown_path=data/helena/spectral_crowns/crowns.parquet
#save_path=data/helena/features/
#IDcolumn=UniqueID
#n_jobs=10
#for year in 2018  2020  2022
#do
#python3 src/make_features.py --tif_path=${tif_path}${year}/${year}.vrt --crown_path=$crown_path --save_path=$save_path --year=$year --IDcolum=$IDcolumn --n_jobs=$n_jobs --chunk_size=2048
#done


from pathlib import Path
import math
import joblib
from joblib import Parallel, delayed, dump, load
import contextlib
import os
from shutil import rmtree

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
from tqdm import tqdm
import argparse


def parse_arguments():
    '''parses the arguments, returns args'''

    # init parser
    parser = argparse.ArgumentParser()

    # add args
    parser.add_argument(
        '--tif_path',
        type=str,
        required=True,
        help='Path to spectral imagery.'
    )

    parser.add_argument(
        '--crown_path',
        type=str,
        required=True,
        help='Path to crowns parquet.'
    )
    
    parser.add_argument(
        '--save_path',
        type=str,
        required=True,
        help='Path where output will be saved.'
    )
    
    parser.add_argument(
        '--year',
        type=str,
        required=True,
        help='Year of data.'
    )
    
    parser.add_argument(
        '--IDcolumn',
        type=str,
        required=True,
        help='Name of column to be used as IDs for crowns.'
    )
    
    parser.add_argument(
        '--n_jobs',
        type=int,
        required=False,
        default=-2,
        help='''Maximum number of concurrently running jobs.
        Defaults to -2, which means all but 1 CPUs are used.'''
    )
    
    parser.add_argument(
        '--label',
        type=str,
        required=False,
        help='Optional - name of column containing labels for classified crowns.'
    )
    
    parser.add_argument(
        '--cols',
        type=str,
        required=False,
        help='''Optional - column names to read, defaults to ['UniqueID', 'treatment', 'geometry']''',
        default=['UniqueID', 'treatment', 'geometry']
    )
    
    parser.add_argument(
        '--chunk_size',
        type=int,
        required=False,
        help='Optional - number of crowns to process per chunk, default=4096',
        default=4096
    )
    

    # parse the args
    args = parser.parse_args()
    
    # make paths in to Paths
    args.tif_path = Path(args.tif_path)
    args.crown_path = Path(args.crown_path)
    args.save_path = Path(args.save_path)

    return(args)

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    '''
    Context manager to patch joblib to report into
    tqdm progress bar given as argument
    '''
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close() 
        

def inner_func(row, xa, masked_rgi, masked_ndvi, r_,
               g_, b_, n_, IDcolumn, label, bins):
    '''guts of make_model_input to run in parallel'''
    
    # this is f!#@ed up, but Parallel and the
    # context manager make it necessary for some reason
    import rioxarray
    
    # calculate luminosity fractions
    lum = xa.luminosity.rio.clip([row.geometry]).to_numpy().flatten()
    lum_tot = lum.shape[0]
    lum_fracs = [((lum < f).sum() - (lum < f - 0.1).sum()) / lum_tot for f in bins]

    # calculate rgi fracs
    rgi = masked_rgi.rio.clip([row.geometry]).to_numpy().flatten()
    rgi = rgi[~np.isnan(rgi)]
    rgi_tot = len(rgi)
    if rgi_tot == 0:
        rgi_fracs = [-99] * 10
    else:
        rgi_fracs = [((rgi < f).sum() - (rgi < f - 0.1).sum()) / rgi_tot for f in bins]
        
    # and normed pix colr fracs
    r = r_.rio.clip([row.geometry]).to_numpy().flatten()
    r = r[~np.isnan(r)]
    c_tot = len(r)
    
    g = g_.rio.clip([row.geometry]).to_numpy().flatten()
    g = g[~np.isnan(g)]

    b = b_.rio.clip([row.geometry]).to_numpy().flatten()
    b = b[~np.isnan(b)]

    n = n_.rio.clip([row.geometry]).to_numpy().flatten()
    n = n[~np.isnan(n)]

    if c_tot == 0:
        r_fracs = [-99] * 10
        g_fracs = [-99] * 10
        b_fracs = [-99] * 10
        n_fracs = [-99] * 10
    else:
        r_fracs = [((r < f).sum() - (r < f - 0.1).sum()) / c_tot for f in bins]
        g_fracs = [((g < f).sum() - (g < f - 0.1).sum()) / c_tot for f in bins]
        b_fracs = [((b < f).sum() - (b < f - 0.1).sum()) / c_tot for f in bins]
        n_fracs = [((n < f).sum() - (n < f - 0.1).sum()) / c_tot for f in bins]
                
    # calculate means and stdevs
    if rgi_tot == 0:
        ndvi_mean, ndvi_std = -99, -99
        rgi_mean, rgi_std = -99, -99
        savi_mean, savi_std = -99, -99
        r_mean, r_std = -99, -99
        g_mean, g_std = -99, -99
        b_mean, b_std = -99, -99
        n_mean, n_std = -99, -99
    else:
        #NOTE: .values * 1 casts 1 item DataArray to float
        ndvi_mean = masked_ndvi.mean(skipna=True).values * 1
        ndvi_std = masked_ndvi.std(skipna=True).values * 1

        rgi_mean = rgi.mean()
        rgi_std = rgi.std()

        savi_mean = xa.SAVI.mean(skipna=True).values * 1
        savi_std = xa.SAVI.std(skipna=True).values * 1

        # ensure no infs
        if math. isinf(ndvi_mean):
            raise Exception('ndvi_mean is inf')
        if math. isinf(ndvi_std):
            raise Exception('ndvi_std is inf')
        if math. isinf(savi_mean):
            raise Exception('ndvi_mean is inf')
        if math. isinf(savi_std):
            raise Exception('ndvi_std is inf')


        r_mean = r.mean()
        r_std = r.std()

        g_mean = g.mean()
        g_std = g.std()

        b_mean = b.mean()
        b_std = b.std()

        n_mean = n.mean()
        n_std = n.std()

    if label is None:
        row[label] = -99

    return (
        [row[IDcolumn], (row[label] + 1) / 2] +
        lum_fracs +
        rgi_fracs + 
        r_fracs + 
        g_fracs + 
        b_fracs + 
        n_fracs +
        [ndvi_mean, ndvi_std, rgi_mean, rgi_std, savi_mean, savi_std] +
        [r_mean, r_std, g_mean, g_std, b_mean, b_std, n_mean, n_std]
        )
    

def make_model_inputs(
    crown_path,
    cols,
    tif_path,
    save_path,
    y,
    IDcolumn,
    n_jobs,
    chunk_size,
    label=None
    ):
    '''
    Returns DataFrame with features for use in classification model.
    The resulting DataFrame has 'ID' column which matches that in crowns.
    The DataFrame also has a 'label' column, see params for more detail.  

    params:
        crowns   - str - path to OGR readable vector file containing tree crowns.
        xa      - xr data array - image used in producing features, already read
                         with rioxarray.
        label    - str - specifies column containing labels.  If specified 'label'
                         column in resulting DataFrame will contain contents of 
                         specified column. Otherwise 'label' column contain -99.
        IDcolumn - str - column to use as matching ID with crowns
    '''
    # process in chunks (unfortunately no chunksize arg in read_parquet)
    n = len(gpd.read_parquet(crown_path)[args.cols])
    chunks = list(np.arange(chunk_size, n, chunk_size))
    if chunks[-1] != n:
        chunks = chunks + [n]

    for i, j in enumerate(np.arange(0, n, chunk_size)):
        if j == 0:
            previous = j
            continue
        
        # make path to output file
        dst = save_path / f'features_{y}_{i}.parquet'

        # if the output file already exists, skip chunk
        if dst.is_file():
            continue
        
        print(f'\t\t\t-- on {i} of {len(chunks)} --')
        # get the extent of the crowns
        xmin, ymin, xmax, ymax = gpd.read_parquet(crown_path).iloc[previous:j, :][cols].total_bounds

        # clip the image
        xa = rioxarray.open_rasterio(tif_path).astype(np.float32).rio.clip_box(
            minx=xmin,
            miny=ymin,
            maxx=xmax,
            maxy=ymax
        ).to_dataset(name='band_data')

        # normalized the band_data
        print(f'\t\t--normalizing (step 1/7)...')
        band_data = xa.band_data.to_numpy().astype(np.float16)
        band_data = (band_data - np.nanmin(band_data)) * (255 / (np.nanmax(band_data) - np.nanmin(band_data)))

        # calculate relative greenness
        print(f'\t\t--calculating RGI (step 2/7)...')
        red = band_data[0]
        green = band_data[1]
        blue = band_data[2]
        nir = band_data[3]
        rgi = green / (red + green + blue)
        xa['rgi'] = (('y', 'x'), rgi)

        # calculate pixel by pixel normalized R, G, B, and NIR
        print(f'\t\t--pix norming (step 3/7)...')
        rgbn_tot = red + green + blue + nir
        xa['red_'] = (('y', 'x'), red  / rgbn_tot)
        xa['blue_'] = (('y', 'x'), blue  / rgbn_tot)
        xa['green_'] = (('y', 'x'), green  / rgbn_tot)
        xa['nir_'] = (('y', 'x'), nir  / rgbn_tot)

        # calculate NDVI and SAVI
        print(f'\t\t--NDVI and SAVI (step 4/7)...')
        nir_agg = xa.band_data[3].astype(float)
        red_agg = xa.band_data[2].astype(float)
        ndvi_agg = (nir_agg - red_agg) / (nir_agg + red_agg)
        L = 1.0
        savi_agg = (1 + L) * (nir_agg - red_agg) / (nir_agg + red_agg + L)
        xa['NDVI'] = ndvi_agg
        xa['SAVI'] = savi_agg
        
        del nir_agg, red_agg, ndvi_agg, savi_agg

        # calculate RGB luminosity
        print(f'\t\t--luminosity (step 5/7)...')
        luminosity = band_data[:3].mean(axis=0) / 255
        xa['luminosity'] = (('y', 'x'), luminosity)

        # mask out shadows and soil for RGI,NDVI, and normed pix colors
        print(f'\t\t--masking (step 6/7)...')
        mask = (luminosity > 0.176) & (luminosity < 0.569) 
        masked_rgi = xa.rgi.where(mask)
        masked_ndvi = xa.NDVI.where(mask)
        r_ = xa.red_.where(mask)
        g_ = xa.green_.where(mask)
        b_ = xa.blue_.where(mask)
        n_ = xa.nir_.where(mask)
        
        print(f'\t\t--adding index data (step 7/7)...')
        
        # read crowns
        crowns = gpd.read_parquet(crown_path).iloc[previous:j, :][cols]
        
        # make disk storage
        mem_dir = Path.cwd() / 'joblib_memmap'
            
        # make sure dir is not there to avoid
        # OSError: [Errno 24] Too many open files
        try:
            os.mkdir(mem_dir)
        except FileExistsError:
            rmtree(mem_dir)
            os.mkdir(mem_dir)
        
        memmap_path0 = mem_dir / 'data_memmap0'
        dump(xa, memmap_path0)
        xa = load(memmap_path0, mmap_mode='r')
        
        memmap_path1 = mem_dir / 'data_memmap1'
        dump(masked_rgi, memmap_path1)
        masked_rgi = load(memmap_path1, mmap_mode='r')
        
        memmap_path2 = mem_dir / 'data_memmap2'
        dump(masked_ndvi, memmap_path2)
        masked_ndvi = load(memmap_path2, mmap_mode='r')
        
        memmap_path3 = mem_dir / 'data_memmap3'
        dump(r_, memmap_path3)
        r_ = load(memmap_path3, mmap_mode='r')
        
        memmap_path4 = mem_dir / 'data_memmap4'
        dump(g_, memmap_path4)
        g_ = load(memmap_path4, mmap_mode='r')
        
        memmap_path5 = mem_dir / 'data_memmap5'
        dump(b_, memmap_path5)
        b_ = load(memmap_path5, mmap_mode='r')
        
        memmap_path6 = mem_dir / 'data_memmap6'
        dump(n_, memmap_path6)
        n_ = load(memmap_path6, mmap_mode='r')
        
        # attach values to crowns
        bins = np.arange(0.1, 1.1, 0.1)
        with tqdm_joblib(tqdm(desc='Calculating features', total=len(crowns))) as progress_bar:
            data = Parallel(n_jobs=n_jobs)(
                delayed(inner_func)
                (
                    row,
                    xa,
                    masked_rgi,
                    masked_ndvi,
                    r_,
                    g_,
                    b_,
                    n_,
                    IDcolumn,
                    label,
                    bins
                )
                for _, row in crowns.iterrows()
            )
        

        feat_cols = [
            IDcolumn, 'label',
            'lum10', 'lum20', 'lum30', 'lum40', 'lum50',
            'lum60' ,'lum70', 'lum80', 'lum90', 'lum100',
            'rgi10', 'rgi20', 'rgi30', 'rgi40', 'rgi50',
            'rgi60' ,'rgi70', 'rgi80', 'rgi90', 'rgi100',
            'r10', 'r20', 'r30', 'r40', 'r50',
            'r60' ,'r70', 'r80', 'r90', 'r100',
            'g10', 'g20', 'g30', 'g40', 'g50',
            'g60' ,'g70', 'g80', 'g90', 'g100',
            'b10', 'b20', 'b30', 'b40', 'b50',
            'b60' ,'b70', 'b80', 'b90', 'b100',
            'n10', 'n20', 'n30', 'n40', 'n50',
            'n60' ,'n70', 'n80', 'n90', 'n100',
            'ndvi_mean', 'ndvi_std',
            'rgi_mean', 'rgi_std',
            'savi_mean', 'savi_std',
            'r_mean', 'r_std',
            'g_mean', 'g_std',
            'b_mean', 'b_std',
            'n_mean', 'n_std'
        ]

        data = pd.DataFrame(data, columns=feat_cols)
        data.to_parquet(dst)
        print(y, 'saved to ', str(dst))
        
        # cleanup
        try:
            del data
            rmtree(mem_dir)
        except:
            # no reason to get in a tizzy if this fails
            pass
        
        previous = j
        

if __name__ == '__main__':
    # parse args
    args = parse_arguments()
    print(f'-- {args.year} --')
    
    
    
    # make inputs
    make_model_inputs(
        args.crown_path,
        args.cols,
        args.tif_path,
        args.save_path,
        args.year,
        args.IDcolumn,
        args.n_jobs,
        args.chunk_size,
        label=args.label
        )
# %%
