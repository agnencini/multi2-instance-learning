import os
import pickle
import numpy as np
import scipy.misc as misc
from random import randint, uniform
from keras.utils import np_utils

# Represents a rectangle. Used when generating data to store digits
# location and check overlaps.
class Rectangle:

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def overlap(self, r):
        return not (self.x1 > r.x2 or r.x1 > self.x2 or
                    self.y1 > r.y2 or r.y1 > self.y2)

# Data structure that store a MNIST dataset (training or testing), with
# both categorical and plain number labels.
class MNISTDataset:

    def __init__(self, data, labels, plain):
        self.data = np.ndarray((len(data), 28, 28))
        self.labels = np.ndarray((len(labels), 10))
        self.plain = plain

        for i in range(len(data)):
            self.data[i] = data[i]
            self.labels[i] = labels[i]

    def size(self):
        return self.data.shape[0]

    def get(self, index):
        return (self.data[index], self.labels[index], self.plain[index])

    # Get a random sample of a specific class.
    def get_random_sample(self, label):

        while True:
            index = randint(0, self.size() - 1)
            data, label_, plain = self.get(index)

            if plain == label:
                return data

# Create MNIST dataset directly from images (folder MNIST). The dataset
# is stored in a MNISTDataset object. One can select training or testing.
def create_MNIST_dataset(type='testing'):

    data = []
    # Categorical labels (vectors)
    labels = []
    # Plain labels (just numbers)
    plain = []
    for i in range(10):
        for filename in os.listdir('MNIST/' + type + '/' + str(i)):
            image = misc.imread('MNIST/' + type + '/' + str(i) + '/' + filename)
            data.append(image)
            labels.append(np_utils.to_categorical(i, num_classes=10))
            plain.append(i)

    dataset = MNISTDataset(data, labels, plain)
    return dataset

# Create a random captcha-like image from MNIST numbers. Digits are random
# placed and random scaled on a black background. Return both the image and
# the categorical label.
def create_captcha(mnist, numbers, label, classes,
                    scale_range=(0.75, 1.25), image_size=(200, 200)):

    captcha = np.zeros(image_size)

    rects = []
    counter = 0
    while len(rects) < len(numbers):

        # Random scaling
        factor = uniform(scale_range[0], scale_range[1])
        w = int(28 * factor)
        h = int(28 * factor)

        # Random positioning
        x = randint(0, image_size[0] - w - 1)
        y = randint(0, image_size[1] - h - 1)

        # Check rectangle overlapping, drop in case
        r = Rectangle(x, y, x + w, y + h)
        overlap = False
        for rect in rects:
            if rect.overlap(r):
                overlap = True
                break

        if overlap:
            continue

        # Choose a random digit from the list of desired numbers.
        image = mnist.get_random_sample(numbers[counter])
        image = misc.imresize(image, (w, h))
        rects.append(r)

        # Draw the number to the background
        for i in range(w):
            for j in range(h):
                if image[i][j] != 0:
                    captcha[i + r.x1][j + r.y1] = image[i][j]
        counter += 1

    return captcha, np_utils.to_categorical(label, num_classes=classes)

# Create a dataset (samples and labels) of captcha-like images. The dataset has
# 2 classes: positive images contain at least one instance of a key digit.
def create_dataset(positive_number, num_samples, num_digits, scale_range=(0.5, 0.5), image_size=(56, 56), data_type='testing', save_PNG = False):

	mnist = None
	if data_type == 'testing':
		mnist = pickle.load(open('mnist-testing.dat', 'rb'))
	else:
		mnist = pickle.load(open('mnist-training.dat', 'rb'))

	samples_per_class = num_samples // 2
	data = np.ndarray((num_samples, image_size[0], image_size[1]))
	labels = np.ndarray((num_samples, 2))

	# Build positive samples
	counter = 0
	for i in range(samples_per_class):

		numbers = [positive_number]
		for j in range(num_digits - 1):
			numbers.append(randint(0, 9))

		im, lbl = create_captcha(mnist, numbers, 0, 2, scale_range, image_size)
		data[counter] = im
		labels[counter] = lbl
		counter += 1

    # Build negative samples
	for i in range(samples_per_class):

		numbers = []
		while(len(numbers) != num_digits):
			n = randint(0, 9)
			if n != positive_number:
				numbers.append(n)

		im, lbl = create_captcha(mnist, numbers, 1, 2, scale_range, image_size)
		data[counter] = im
		labels[counter] = lbl
		counter += 1

	if save_PNG:
		for i in range(num_samples):
			misc.imsave('data/' + 'class' + str(np.argmax(labels[i])) + '-' + str(i) + '.png', data[i])

	return data, labels

# Store all samples and labels from both training and testing sets
def store_datasets(trainx, trainy, testx, testy):

    with open('datasets/trainx.dat', 'wb') as f:
        pickle.dump(trainx, f)
    with open('datasets/trainy.dat', 'wb') as f:
        pickle.dump(trainy, f)
    with open('datasets/testx.dat', 'wb') as f:
        pickle.dump(testx, f)
    with open('datasets/testy.dat', 'wb') as f:
        pickle.dump(testy, f)
