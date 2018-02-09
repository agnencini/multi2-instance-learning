import numpy as np
from data_generation import Rectangle

# Extract patches from a grayscale image. Utility function used by function
# get_multi3_patches below.
def extract_patches_(image, patch_size, stride):

    patches = []
    for i in range(0, image.shape[0] - patch_size[0], stride):
        for j in range(0, image.shape[1] - patch_size[1], stride):
            a = image[i:i+patch_size[0], j:j+patch_size[1]]
            patches.append(a)
    return patches

# Extract 3 levels of patches from a batch of grayscale images.
def get_multi3_patches(batch, bp_size, mp_size, sp_size, stride1, stride2, stride3):

    bp_sample = extract_patches_(batch[0], bp_size, stride1)
    n_bp = len(bp_sample)

    mp_sample = extract_patches_(bp_sample[0], mp_size, stride2)
    n_mp = len(mp_sample)

    sp_sample = extract_patches_(mp_sample[0], sp_size, stride3)
    n_sp = len(sp_sample)

    result = np.ndarray((batch.shape[0], n_bp, n_mp, n_sp, sp_size[0] * sp_size[1]))
    for b in range(batch.shape[0]):
        bp = extract_patches_(batch[0], bp_size, stride1)

        for i, bp_ in enumerate(bp):
            mp = extract_patches_(bp_, mp_size, stride2)

            for j, mp_ in enumerate(mp):
                sp = extract_patches_(mp_, sp_size, stride3)

                for k, sp_ in enumerate(sp):
                    result[b, i, j, k] = sp_.flatten()

    return result
