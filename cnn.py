import pickle
import numpy as np

import keras
from keras.layers import Conv2D, Input, Dense, MaxPooling2D, Maximum, Flatten, Dropout, Activation, BatchNormalization
from keras.models import Model, Sequential
from keras.utils import to_categorical

from data_generation import *
from train import stats, log

# A CNN model that is used to make comparisons with the bag layer approach.
def get_cnn_model(input_shape_):

    model = Sequential()
    model.add(Conv2D(32, (5,5), activation='relu', input_shape=input_shape_))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    return model

trainx = pickle.load(open('datasets/trainx.dat', 'rb'))
trainy = pickle.load(open('datasets/trainy.dat', 'rb'))
trainx = np.reshape(trainx, (trainx.shape[0], trainx.shape[1], trainx.shape[2], 1))

testx = pickle.load(open('datasets/testx.dat', 'rb'))
testy = pickle.load(open('datasets/testy.dat', 'rb'))
testx = np.reshape(testx, (testx.shape[0], testx.shape[1], testx.shape[2], 1))

model = get_cnn_model((trainx.shape[1], trainx.shape[2], trainx.shape[3]))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

indices = np.arange(len(trainx))
batch_size = 20
n_batches = len(indices) // batch_size

for e in range(20):
    print('Epoch ' + str(e + 1))
    np.random.shuffle(indices)

	# Training section
    for b in range(n_batches):

        if((b + 1) % 10 == 0):
            print('Batch ' + str(b + 1) + '/' + str(n_batches))

        start_batch = b * batch_size
        end_batch = min((b + 1) * batch_size, len(indices))
        current_indices = indices[start_batch : end_batch]

        x_batch = trainx[current_indices]
        y_batch = trainy[current_indices]
        model.train_on_batch(x_batch, y_batch)

        if((b + 1) % 10 == 0):
            metrics = model.test_on_batch(x_batch, y_batch)
            print('LOSS ' + str(metrics[0]) + ' ACC. ' +  str(metrics[1]))

            response = model.predict_on_batch(x_batch)
            tp, tn, fp, fn = stats(response, y_batch)
            print('tp', tp, 'tn', tn, 'fp', fp, 'fn', fn)

	# Testing/validation section
    accuracy = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    print('Start validation of epoch ' + str(e))
    for k in range(0, testx.shape[0], 10):

        test_x_batch = testx[k:k+10]
        test_y_batch = testy[k:k+10]

        metrics = model.test_on_batch(test_x_batch, test_y_batch)
        print('Batch ' + str(k // 10) + '/' + str(int(testx.shape[0] // 10)) + ' (Loss, Acc.)=(' + str(metrics[0]) + ', ' + str(metrics[1]) + ')')
        accuracy += float(metrics[1])

        response = model.predict_on_batch(test_x_batch)
        tp_, tn_, fp_, fn_ = stats(response, test_y_batch)
        tp += tp_
        tn += tn_
        fp += fp_
        fn += fn_

    accuracy /= float(testx.shape[0] // 10)
    precision = float(tp) / float(tp + fp)
    recall = float(tp) / float(tp + fn)
    log('Precision ' + str(precision))
    log('Recall ' + str(recall))
    log('TP ' + str(tp) + ' TN ' + str(tn) + ' FP ' +str(fp) + ' FN ' + str(fn))
    log('Accuracy ' + str(accuracy))
