#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dyra-Net with Keras Functional-API
Task: Classification
Approach: Dyad Ranking
Data: Fashion-MNIST

@author: dschaefer (June 2018)
"""
import time
import pickle

import keras
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Input, BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Model
from keras.constraints import maxnorm
from keras import backend as K
from keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np

from DyraNetClassificationEvalCallback import DyraNetClassificationEvalCallback
from PrintMetricsCallback import PrintMetricsCallback
from utils.mnist_reader import load_mnist_idx_from_folder
from utils.evaluation import predict_class

# %% Settings
batch_size = 128
nb_classes = 10
nb_epochs = 50
img_rows, img_cols = 28, 28
nb_filters = 32
pool_size = 2
kernel_size = 3
keras.backend.image_dim_ordering()

x_input_shape = (img_rows, img_cols, 1)  # TensorFlow ordering
y_input_shape = (nb_classes,)

# %% Load dataset
data_dir = 'data/raw'
data = load_mnist_idx_from_folder(data_dir)

# %% Data preparation
X_train = data['train_imgs']
X_test = data['test_imgs']
y_train = data['train_labels']
y_test = data['test_labels']
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = np.expand_dims(X_train, axis=1).astype('float32')
X_test = np.expand_dims(X_test, axis=1).astype('float32')

# Normalize image features 
X_train /= 255.0
X_test /= 255.0

# %% Create Tr/Val split
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=111)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


# %% Ranking loss (MLE)
def pairwise_ranking_log_loss(y_true, y_pred):
    """
        Ranking loss (Bradley-Terry Log Loss)
    """
    loss = tf.convert_to_tensor(0, dtype=tf.float32)
    for i in range(0, 2 * (nb_classes - 1) * batch_size, 2):
        try:
            u1 = y_pred[i + 0]
            u2 = y_pred[i + 1]
            loss = (loss + K.log(K.exp(u1) + K.exp(u2)) - u1)
        except:
            continue
    loss = loss / ((nb_classes - 1) * batch_size)
    return loss


# %%
def pair_generator(x_features, label_features, batch_size, nb_classes):
    """
        Data generator for keras model fit procedure.
    
        The generator produces 2*(nb_classes-1)*batch_size 
        pairwise preferences from (instance, class label) pairs,
        where a class label is one of nb_classes.
        
        Args:
            x_features - instance features
            label_features - class labels as 1-of-k vectors
            batch_size - batch size in terms of number of instance, label pairs
            
        The resulting number of training examples, however, is not batch_size, but
        2*(nb_classes-1)*batch_size, because a single instance, class pair
        is converted into nb_classes-1 preferences.
        
        Example: a,b = pair_generator(X_train, y_train, 1, 10)
    """
    nb_gen_examples = 2 * (nb_classes - 1) * batch_size
    batch_x_features = np.zeros((nb_gen_examples, img_rows, img_cols, 1))
    batch_y_features = np.zeros((nb_gen_examples, nb_classes))
    batch_pseudo_labels = np.zeros(nb_gen_examples)

    while True:
        cnt2 = 0
        for i0 in range(batch_size):
            index = np.random.choice(len(x_features), 1)
            feat = x_features[index]
            winning_y_features = label_features[index]
            winning_class = np.where(label_features[index][0] == 1)[0].item()
            residual_y_features = np.zeros((nb_classes - 1, nb_classes))
            cnt = 0
            for i1 in range(nb_classes):
                if i1 != winning_class:
                    residual_y_features[cnt][i1] = 1
                    cnt = cnt + 1

            for i1 in range(nb_classes - 1):
                batch_x_features[cnt2] = feat
                batch_y_features[cnt2] = winning_y_features
                batch_x_features[cnt2 + 1] = feat
                batch_y_features[cnt2 + 1] = residual_y_features[i1]
                cnt2 = cnt2 + 2
        yield ([batch_x_features, batch_y_features], batch_pseudo_labels)


# %%
def model_architecture(nb_filters, kernel_size, x_input_shape, y_input_shape):
    """
        Definition of a Dyra-Net model instance using Keras functional API
    """
    x_input_shape = x_input_shape
    y_input_shape = y_input_shape

    x_input = Input(shape=x_input_shape)
    x_conv1 = Convolution2D(filters=nb_filters, kernel_size=(kernel_size, kernel_size), activation='relu')(x_input)
    x_pool1 = MaxPooling2D(pool_size=(pool_size, pool_size))(x_conv1)
    x_bn1 = BatchNormalization()(x_pool1)
    x_conv2 = Convolution2D(filters=nb_filters, kernel_size=(kernel_size, kernel_size), activation='relu')(x_bn1)
    x_pool2 = MaxPooling2D(pool_size=(pool_size, pool_size))(x_conv2)
    x_bn2 = BatchNormalization()(x_pool2)
    x_drop1 = Dropout(rate=0.25)(x_bn2)
    x_flat = Flatten()(x_drop1)

    y_input = Input(shape=y_input_shape)
    y_fc1 = Dense(units=32, activation='tanh')(y_input)

    xy_cat = concatenate([x_flat, y_fc1])
    xy_fc1 = Dense(units=128 + 32, activation='tanh', kernel_constraint=maxnorm(3))(xy_cat)
    xy_fc1_drop = Dropout(rate=0.5)(xy_fc1)
    xy_fc2 = Dense(units=1, activation='linear', kernel_constraint=maxnorm(3))(xy_fc1_drop)
    model = Model(inputs=[x_input, y_input], outputs=xy_fc2)
    model.compile(loss=pairwise_ranking_log_loss, optimizer='adam')

    return model


# %% Training
model = model_architecture(nb_filters, kernel_size, x_input_shape, y_input_shape)
print(model.summary())

# Config
train_steps_per_epoch = int(X_train.shape[0] / batch_size)
validation_steps_per_epoch = int(X_val.shape[0] / batch_size)

print_metrics_callback = PrintMetricsCallback()
accuracy_callback = DyraNetClassificationEvalCallback(X_val, np.argmax(y_val, axis=1))
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=25, verbose=1, mode='auto')

# Training
start_time = time.time()
hist = model.fit_generator(pair_generator(X_train, y_train, batch_size, nb_classes),
                           steps_per_epoch=train_steps_per_epoch,
                           epochs=nb_epochs,
                           callbacks=[print_metrics_callback, accuracy_callback, early_stopping_callback],
                           validation_data=pair_generator(X_val, y_val, batch_size, nb_classes),
                           validation_steps=validation_steps_per_epoch)
end_time = time.time()
model.save('/output/dyranet_model.h5')
with open('/output/historyDict', 'wb') as file_pi:
    pickle.dump(hist.history, file_pi)
print('Training time: {:4.3f} minutes'.format((end_time - start_time) / 60.0))

# %% Testing
num_te_examples = len(y_test)
te_accuracy = 0
for i in range(num_te_examples):
    idx = i
    te_inst = X_test[idx]
    true_cl = np.where(y_test[idx] == 1)[0].item() + 1
    pred_cl, scores = predict_class(model, te_inst, 10)
    if true_cl == pred_cl:
        te_accuracy = te_accuracy + 1
te_accuracy = te_accuracy / num_te_examples
print('-------------------------------------')
print('Classification Accuracy (Test-Set): {} '.format(te_accuracy))
print('-------------------------------------')
