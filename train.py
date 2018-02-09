import pickle
import datetime
import numpy as np
from keras.optimizers import Adam

from data_generation import *
from models import *
from patching_dense import *
from patching_cc import *
from patching_multi3 import *

VALID_PATCHING_MODES = ['dense', 'cc', 'multi3']
PATCHING_MODE = 'dense'

# Adam parameters
LR = 0.001
BETA1 = 0.9
BETA2 = 0.999
DECAY = 0.0

# Dense patching parameters
tb_size   = [15, 15]
sb_size   = [5, 5]
tb_stride = [5, 5]

# CC patching parameters
cc_tb_size = (15, 15)
cc_sb_size = (5, 5)
cc_tb_max  = 40
cc_stride  = 1

# Multi3 patching parameters
m3_tb_size = (28, 28)
m3_mb_size = (14, 14)
m3_sb_size = (7, 7)
m3_stride1 = 7
m3_stride2 = 4
m3_stride3 = 1

def log(text):
	message = '[' + str(datetime.datetime.now()) + ']' + '\t' + text + '\n'
	print(message)

	with open('log.txt', 'a') as logfile:
		logfile.write(message + '\n')

# Compute statistics (true positives, true negatives, false postives,
# false negatives)
def stats(prediction, y_batch):

	tp = tn = fp = fn = 0
	for i in range(prediction.shape[0]):
		predicted = np.argmax(prediction[i])
		label = np.argmax(y_batch[i])

		if predicted == 0 and predicted == label:
			tp += 1
		elif predicted == 1 and predicted == label:
			tn += 1
		elif predicted == 0 and predicted != label:
			fp += 1
		else:
			fn += 1

	return tp, tn, fp, fn

if __name__ == '__main__':

	# Load datasets
	trainx = pickle.load(open('datasets/trainx.dat', 'rb'))
	trainy = pickle.load(open('datasets/trainy.dat', 'rb'))
	trainx = np.reshape(trainx, (trainx.shape[0], trainx.shape[1], trainx.shape[2], 1))

	testx = pickle.load(open('datasets/testx.dat', 'rb'))
	testy = pickle.load(open('datasets/testy.dat', 'rb'))
	testx = np.reshape(testx, (testx.shape[0], testx.shape[1], testx.shape[2], 1))

	# Get an appropriate shaped model based on the chosen patching mode
	X_sample = extend_images(trainx[0:1], tb_size)
	mask_tb = create_subsampling_mask(patches_shape=tb_size, img_shape=X_sample.shape[1:-1], stride=tb_stride)
	if PATCHING_MODE == 'dense':
		sample = get_patches(extend_images(trainx[0:10], tb_size), tb_size, sb_size, mask_tb)
		model = get_model((sample.shape[1], sample.shape[2], sample.shape[3]))
	elif PATCHING_MODE == 'cc':
		sample = get_cc_patches(trainx[0:10], cc_tb_size, cc_sb_size, cc_tb_max, cc_stride)
		model = get_model((sample.shape[1], sample.shape[2], sample.shape[3]))
	else:
		sample = get_multi3_patches(trainx[0:10], m3_tb_size, m3_mb_size, m3_sb_size, m3_stride1, m3_stride2, m3_stride3)
		model = get_multi3_model((sample.shape[1], sample.shape[2], sample.shape[3], sample.shape[4]))

	adam = Adam(lr=LR, beta_1=BETA1, beta_2=BETA2, decay=DECAY)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
	indices = np.arange(len(trainx))
	batch_size = 10
	epochs = 10
	n_batches = len(indices) // batch_size

	log('Start training' + ', patching=' + PATCHING_MODE + ', samples=' + str(trainx.shape[0]))
	for e in range(epochs):
		log('Epoch ' + str(e + 1))
		np.random.shuffle(indices)

		for b in range(n_batches):

			if((b + 1) % 10 == 0):
				log('Batch ' + str(b + 1) + '/' + str(n_batches))

			start_batch = b * batch_size
			end_batch = min((b + 1) * batch_size, len(indices))
			current_indices = indices[start_batch : end_batch]

			if PATCHING_MODE == 'dense':
				x_batch = get_patches(extend_images(trainx[current_indices], tb_size), tb_size, sb_size, mask_tb)
			elif PATCHING_MODE == 'cc':
				x_batch = get_cc_patches(trainx[current_indices], cc_tb_size, cc_sb_size, cc_tb_max, cc_stride)
			else:
				x_batch = get_multi3_patches(trainx[current_indices], m3_tb_size, m3_mb_size, m3_sb_size, m3_stride1, m3_stride2, m3_stride3)

			y_batch = trainy[current_indices]
			model.train_on_batch(x_batch, y_batch)

			# Compute accuracy on batch every 10
			if((b + 1) % 10 == 0):
				metrics = model.test_on_batch(x_batch, y_batch)
				print('LOSS ' + str(metrics[0]) + ' ACC. ' +  str(metrics[1]))#log
				response = model.predict_on_batch(x_batch)
				tp, tn, fp, fn = stats(response, y_batch)
				print('tp', tp, 'tn', tn, 'fp', fp, 'fn', fn)

		# Testing/validation section
		accuracy = 0
		tp = 0
		tn = 0
		fp = 0
		fn = 0
		log('Start validation of epoch ' + str(e + 1))
		for k in range(0, testx.shape[0], 10):

			if PATCHING_MODE == 'dense':
				test_x_batch = get_patches(extend_images(testx[k:k+10], tb_size), tb_size, sb_size, mask_tb)
			elif PATCHING_MODE == 'cc':
				test_x_batch = get_cc_patches(testx[k:k+10], cc_tb_size, cc_sb_size, cc_tb_max, cc_stride)
			else:
				test_x_batch = get_multi3_patches(testx[k:k+10], m3_tb_size, m3_mb_size, m3_sb_size, m3_stride1, m3_stride2, m3_stride3)

			test_y_batch = testy[k:k+10]
			metrics = model.test_on_batch(test_x_batch, test_y_batch)
			log('Batch ' + str(k // 10) + '/' + str(int(testx.shape[0] // 10)) + ' (Loss, Acc.)=(' + str(metrics[0]) + ', ' + str(metrics[1]) + ')')
			accuracy += float(metrics[1])

			response = model.predict_on_batch(test_x_batch)
			tp_, tn_, fp_, fn_ = stats(response, test_y_batch)
			tp += tp_
			tn += tn_
			fp += fp_
			fn += fn_

		# Compute accuracy, precision and recall at the end of epoch validation
		accuracy /= float(testx.shape[0] // 10)
		precision = float(tp) / float(tp + fp)
		recall = float(tp) / float(tp + fn)
		log('Precision ' + str(precision))
		log('Recall ' + str(recall))
		log('TP ' + str(tp) + ' TN ' + str(tn) + ' FP ' +str(fp) + ' FN ' + str(fn))
		log('Accuracy: ' + str(accuracy))
