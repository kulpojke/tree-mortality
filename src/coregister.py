#%%
import cv2
import numpy as np
import xarray as xr
import rioxarray

import os
from pathlib import Path

#%%

template_path = Path('/home/michael/TreeMortality/CHM/10TDL0458245240.tif')
data_paths = [
    Path(f'/home/michael/TreeMortality/Planet/10TDL0458245240/2019{d}/composite.tif')
    for d in ['07', '10']
]

image_dir = Path('/home/michael/TreeMortality/Planet/10TDL0458245240')

#%%
def get_gradient(im) :
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im, cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im, cv2.CV_32F,0,1,ksize=3)
    # Combine the two gradients
    grad = cv2.addWeighted(
        np.absolute(grad_x),
        0.5,
        np.absolute(grad_y),
        0.5,
        0)
    return grad


def scale_template_to_target(template_path, img):
    lidar_dset = rioxarray.open_rasterio(template_path)
    # get intensity and lidar intensity
    image_band0 = img[0].to_numpy().astype(np.float32)
    image_band0[image_band0 == -99] = np.nan
    template = lidar_dset.to_numpy().astype(np.float32)
    # invert template
    inverted_template = template.max() - template
    # set template to scale of red band
    mx = np.nanmax(image_band0)
    mn = np.nanmin(image_band0)
    scaled_template = (
        (inverted_template - inverted_template.min()) *
        abs(mx - mn) /
        abs(inverted_template.max() - inverted_template.min())
        )
    # write inverted scaled template
    dst = template_path.parent / f'{template_path.stem}_inverted_scaled.tif'
    lidar_dset.copy(data=scaled_template).rio.to_raster(dst)


def get_grads(lidar_dset, img):
    # get intensity and lidar intensity
    img_intensity = img[:3].mean(axis=0).to_numpy().astype(np.float32)
    template = lidar_dset.to_numpy().astype(np.float32)
    # set template and img_intensity to same scale
    template = (
        (template - template.min()) *
        abs(img_intensity.max() - img_intensity.min()) /
        abs(template.max() - template.min())
        )
    img_intensity = (img_intensity - img_intensity.min())
    # calculate gradients
    grad_lidar = get_gradient(template)
    grad_img = get_gradient(img_intensity)
    
    return grad_lidar, grad_img


def get_translation(n_iter, termination_eps, template, target):    
    # define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = n_iter
    # define termination criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        number_of_iterations, 
        termination_eps
        )
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(
        template,
        target,
        warp_matrix,
        warp_mode,
        criteria
    )
    
    return cc, warp_matrix


def align_bands(img, warp_matrix):
    aligned = []
    # warp bands
    for i, band in enumerate(img):
        aligned_band = cv2.warpAffine(
            band.to_numpy(),
            warp_matrix,
            (band.shape[1], band.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)     
        aligned.append(aligned_band)
    # stack into np array    
    aligned = np.stack(aligned)
    # put aligned bands into new xr dset
    aligned_img = img.copy(data=aligned)
    
    return aligned_img


def read_notclear(udm_path):
    '''
    returns the clear/not clear band of unusable
    data mask as an boolean np.array, where not_clear = 1.
    '''
    not_clear = rioxarray.open_rasterio(udm_path)[0] == 0
    
    return not_clear


def find_files(image_dir, fname='composite.tif', udm_name='composite_udm2.tif'):
    '''
    Finds all of the files called fname scattered around subdirs of image_dir,
    and matches each file to its udm. Returns a list of tuples (file, udm).
    '''
    images = []
    udms = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file == fname:
                images.append(Path(os.path.join(root, file)))
            if file == udm_name:
                udms.append(Path(os.path.join(root, file)))     
    pairs = [(f, u) for u in udms for f in images if u.parent == f.parent] 
             
    return pairs


def mask_image(image, udm):
    '''Reads image, applies mask.'''
    mask = read_notclear(udm)
    xa = rioxarray.open_rasterio(image)
    date = f'{image.parent.stem[:4]}-{image.parent.stem[4:]}'
    xa = xa.expand_dims(dim={'time': [date]})
    return xa.where(~mask, other=-99)


def make_mean_image(image_dir):
    '''
    TODO: this will need a way to select a time range for the median composite
    '''
    # find files and udms
    pairs = find_files(image_dir)
    # stack up images
    arrs = [mask_image(image, udm) for image, udm in pairs]
    xa = xr.concat(arrs, 'time')
    
    avg = xa.mean(dim='time', skipna=True)
    
    return avg
    
    
    
# make mean image of 2019
xa = make_mean_image(image_dir)

# match with CHM

# match mean image for each year to the mean image for previous/next year


#%%
n_iter = 50000
termination_eps = 1e-10

xa = rioxarray.open_rasterio(template_path)
xa = xa.to_dataset(name='template')

img = make_mean_image(image_dir)

grad_lidar, grad_planet = get_grads(
    xa.rio.set_crs(img.rio.crs).template[0],
    img
    )



cc, warp_matrix = get_translation(
        n_iter,
        termination_eps,
        grad_lidar,
        grad_planet
    )

# align image
aligned_img = align_bands(img, warp_matrix)

# write to tif
dst = image_dir / f'{image_dir.stem}_2019_mean_aligned.tif'
aligned_img.rio.to_raster(dst)

intensity_path = image_dir / f'{image_dir.stem}_2019_mean_intensity_aligned.tif'
aligned_img[:3].mean(axis=0).rio.to_raster(intensity_path)

#%%

xa = rioxarray.open_rasterio(template_path)
xa = xa.to_dataset(name='template')

n_iter = 50000
termination_eps = 1e-15
out_path = Path
for raster_path in data_paths:
    # open image
    img = rioxarray.open_rasterio(raster_path)
    # calculate gradients
    grad_lidar, grad_planet = get_grads(
        xa.rio.set_crs(img.rio.crs).template[0],
        img
        )
    # get translation
    cc, warp_matrix = get_translation(
        n_iter,
        termination_eps,
        grad_lidar,
        grad_planet
    )
    
    # align image
    aligned_img = align_bands(img, warp_matrix)
    
    # write to tifs
    dst = raster_path.parent / f'{raster_path.stem}_aligned.tif'
    aligned_img.rio.to_raster(dst)
    
    intensity_path = raster_path.parent / f'{raster_path.stem}_intensity_aligned.tif'
    aligned_img[:3].mean(axis=0).rio.to_raster(intensity_path)
    
    L = 0.5
    savi = ((1 + L) *
            (aligned_img[3] - aligned_img[0]) /
            (aligned_img[3] + aligned_img[0] + L))
    savi_path = raster_path.parent / f'{raster_path.stem}_SAVI_aligned.tif'
    savi.rio.to_raster(savi_path)
    
 
# %%
