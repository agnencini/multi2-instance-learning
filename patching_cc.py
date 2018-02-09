import cv2
import numpy as np
import scipy.misc as misc
from random import randint
from data_generation import Rectangle

# Get the connected components of an image. Return the bounding boxes and the
# centroids of the connected components.
def get_connected_components(image):

    image_copy = np.array(image, copy=True).astype(np.uint8)
    ret, thresh = cv2.threshold(image_copy, 0, 255, cv2.THRESH_BINARY)

    max_connect = 4
    output = cv2.connectedComponentsWithStats(thresh, max_connect, cv2.CV_32S)

    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]

    rects = [] # List, its length is the number of CCs
    for i in range(1, len(centroids)):

        x0 = stats[i][0]
        y0 = stats[i][1]

        w = stats[i][2]
        h = stats[i][3]

        r = Rectangle(x0, y0, x0 + w, y0 + h)
        rects.append(r)

    # A numpy array of size (n, 2) where n is the number of CCs
    centroids_ = np.ndarray((centroids.shape[0] - 1, centroids.shape[1]))
    for i in range(1, len(centroids)):
        centroids_[i - 1] = centroids[i]

    return rects, centroids_

# Extract 2 levels of patches around the connected components of an image.
def get_cc_patches(batch, bp_size, sp_size, max_bp, sp_stride):

    batch_patches = []
    for i in range(batch.shape[0]):

        image = batch[i]
        rects, centroids = get_connected_components(image)
        patches = []

        counter = 0
        stop = False

        repeatx = 0
        repeaty = 0

        step1 = 0
        for a0 in range(repeatx, image.shape[0] - bp_size[0], 5):
            for b0 in range(repeaty, image.shape[1] - bp_size[1], 5):
                r = Rectangle(a0, b0, a0 + bp_size[0], b0 + bp_size[1])
                overlap = False
                for rect in rects:
                    if r.overlap(rect):
                        overlap = True
                        break

                if overlap:
                    step1 += 1

        stride = (5 * max_bp) // step1
        if stride == 0:
            stride = 1

        while not stop:
            for a0 in range(repeatx, image.shape[0] - bp_size[0], stride):
                if stop:
                    break

                for b0 in range(repeaty, image.shape[1] - bp_size[1], stride):

                    if stop:
                        break

                    r = Rectangle(a0, b0, a0 + bp_size[0], b0 + bp_size[1])
                    overlap = False
                    for rect in rects:
                        if r.overlap(rect):
                            overlap = True
                            break

                    if overlap:
                        patches.append(r)
                        counter += 1

                    if counter >= max_bp:
                        stop = True
                        break

            if stride > 1:
                repeatx += (stride // 2)
            else:
                repeatx += 1
            if repeatx >= image.shape[0] - bp_size[0]:
                repeatx = 0
                if stride > 1:
                    repeaty += (stride // 2)
                else:
                    repeaty += 1
            if repeaty >= image.shape[1] - bp_size[1]:
                repeaty = 0

        # For each big patch, compute small patches
        small_patches_dict = {}
        for j, r in enumerate(patches):
            small_patches = []

            for a in range(r.x1, r.x2 - sp_size[0], sp_stride):
                for b in range(r.y1, r.y2 - sp_size[1], sp_stride):
                    arr = image[a:a+sp_size[0], b:b+sp_size[1]]
                    small_patches.append(arr)
            small_patches_dict[j] = small_patches

        dim1 = len(patches)
        dim2 = len(small_patches_dict[0])
        dim3 = sp_size[0] * sp_size[1]

        def_x = np.ndarray((dim1, dim2, dim3))
        for a in range(dim1):
            for b in range(dim2):
                def_x[a][b] = (small_patches_dict[a])[b].flatten()

        batch_patches.append(def_x)

    # Merge image patches in a single batch
    sample_shape = batch_patches[0].shape
    result = np.ndarray((batch.shape[0], sample_shape[0], sample_shape[1], sample_shape[2]))
    for i in range(batch.shape[0]):
        result[i] = batch_patches[i]
    return result
