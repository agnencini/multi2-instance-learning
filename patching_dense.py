import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.image import PatchExtractor

# Extend an image to fit the patching process.
def extend_images(img, top_bag_size):
    offset_x = np.ceil(img.shape[1]*1./top_bag_size[0]).astype(np.int32) + img.shape[1]
    offset_y = np.ceil(img.shape[2]*1./top_bag_size[1]).astype(np.int32) + img.shape[2]
    ext_img = np.zeros((img.shape[0], offset_x, offset_y, img.shape[3]), dtype=img.dtype)
    ext_img[:,:img.shape[1],:img.shape[2],:] = img
    return ext_img

# Create a mask to extract small patches.
def create_subsampling_mask(patches_shape, img_shape, stride=(5,5)):
    points_x_full = np.arange(0,img_shape[0],1)
    points_x_full = points_x_full[points_x_full < img_shape[0] - patches_shape[0] +1 ]

    points_y_full = np.arange(0,img_shape[1],1)
    points_y_full = points_y_full[points_y_full < img_shape[1] - patches_shape[1] +1 ]

    mask = np.zeros((len(points_x_full),len(points_y_full)))

    points_x = np.arange(0,img_shape[0],stride[0])
    points_x = points_x[points_x < img_shape[0] - patches_shape[0] +1 ]
    points_y = np.arange(0,img_shape[1],stride[1])
    points_y = points_y[points_y < img_shape[1] - patches_shape[1] +1 ]

    for x in points_x:
        for y in points_y:
            mask[x,y] = 1
    return mask > 0

# Extract patches from an image.
def get_patches(img, tb_size, sb_size, mask_tb=None, mask_sb=None):
    tb_extractor = PatchExtractor(tuple(tb_size))
    sb_extractor = PatchExtractor(tuple(sb_size))
    tb_images = tb_extractor.transform(img)
    tb_images = tb_images.reshape((img.shape[0], -1, tb_images.shape[-2], tb_images.shape[-1]))

    if mask_tb is not None:
        tb_images = tb_images[:, mask_tb.ravel(), : ,:]
    tb_images = np.rollaxis(tb_images, 1, 4)
    sb_images = sb_extractor.transform(tb_images)
    sb_images = sb_images.reshape((img.shape[0], -1, sb_images.shape[-3], sb_images.shape[-2], sb_images.shape[-1]))
    sb_images = np.rollaxis(sb_images, 4, 1)

    if mask_sb is not None:
        sb_images = sb_images[:, :, mask_sb.ravel(), : ,:]
    sb_images = sb_images.reshape(list(sb_images.shape[:-2]) + [np.prod(sb_images.shape[-2:])])
    return sb_images
