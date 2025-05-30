#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, cv2, csv
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from utilities import one_hot_encoding, read_train_data, read_label_data, show_train_history
from utilities import build_vgg_model, build_resnet50_model, build_inceptionv3_model
from tensorflow.keras.callbacks import ReduceLROnPlateau

SIZE = 10000
MODEL_FOLDER = "model/"
WIDTH = 140
HEIGHT = 48
IMG_SIZE = WIDTH if WIDTH > HEIGHT else HEIGHT
NUM_DIGIT = 4
PROCESSED_FOLDER = "processed/"
LABEL_CSV_FILE = 'label.csv'
allowedChars = '234579ACFHKMNPQRTYZ'


# In[ ]:


model = build_vgg_model(WIDTH, HEIGHT, allowedChars, NUM_DIGIT)
# model = build_inceptionv3_model(IMG_SIZE, allowedChars, NUM_DIGIT)
# model = build_resnet50_model(IMG_SIZE, allowedChars, NUM_DIGIT)


# In[3]:


print("Reading training data...")

train_data = read_train_data(PROCESSED_FOLDER, SIZE)
train_label = read_label_data(LABEL_CSV_FILE, allowedChars, NUM_DIGIT, SIZE)

print("Reading completed")


# In[ ]:


filepath = MODEL_FOLDER + "{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_digit4_accuracy', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_loss', patience=12, verbose=1, mode='auto')
tensorBoard = TensorBoard(log_dir = 'logs', histogram_freq = 1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, min_lr=1e-6)
callbacks_list = [tensorBoard, earlystop, checkpoint, reduce_lr]


# In[5]:


# model = load_model("thsrc_cnn_model.hdf5")


# In[ ]:


history = model.fit(train_data, train_label, validation_split=0.2, batch_size=4, epochs=80, verbose=1, shuffle=True, callbacks=callbacks_list)


# In[7]:


show_train_history(history, 'digit1_accuracy', 'val_digit1_accuracy')


# In[ ]:




