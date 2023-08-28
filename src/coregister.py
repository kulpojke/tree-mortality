#%%
import cv2
import numpy as np
import xarray as xr
import rioxarray
from pathlib import Path

#%%

template_path = Path('/home/michael/TreeMortality/intensity/10TDL0458245240.tif')
data_paths = [
    Path(f'/home/michael/TreeMortality/Planet/10TDL0458245240/2019{d}/composite.tif')
    for d in ['07', '10']
]
#%%
xa = rioxarray.open_rasterio(template_path)
xa = xa.to_dataset(name='template')


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


def get_grads(lidar_dset, img):

    # get intensity and lidar intensity
    img_intensity = img[:3].mean(axis=0).to_numpy().astype(np.float32)
    lidar_intensity = lidar_dset.template[0].to_numpy().astype(np.float32)
    # set lidar_intensity and img_intensity to same scale
    lidar_intensity = (
        (lidar_intensity - lidar_intensity.min()) *
        abs(img_intensity.max() - img_intensity.min()) /
        abs(lidar_intensity.max() - lidar_intensity.min())
        )
    img_intensity = (img_intensity - img_intensity.min())
    # calculate gradients
    grad_lidar = get_gradient(lidar_intensity)
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



n_iter = 50000
termination_eps = 1e-15
out_path = Path
for raster_path in data_paths:
    # open image
    img = rioxarray.open_rasterio(raster_path)
    # calculate gradients
    grad_lidar, grad_planet = get_grads(
        xa.rio.set_crs(img.rio.crs),
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
    
    # write to tif
    dst = raster_path.parent / (str(raster_path.stem) + '_aligned.tif')
    aligned_img.rio.to_raster(dst)
    
    t_path = raster_path.parent / (str(raster_path.stem) + '_intensity_aligned.tif')
    aligned_img[:3].mean(axis=0).rio.to_raster(t_path)
    
 #%%   
   