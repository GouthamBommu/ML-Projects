#!/usr/bin/env python
# coding: utf-8

#import required packages.
import os
import glob
import numpy as np
import pandas as pd
import sklearn.model_selection
base_dir=os.path.join("C://Users//Goutham//ML//CNN//images//cell_images")
infected_dir=os.path.join(base_dir,"Parasitized")
healthy_dir=os.path.join(base_dir,"Uninfected")
#defining directory
infected_files=glob.glob(infected_dir+"/*.png")
healthy_files=glob.glob(healthy_dir+"/*.png")
len(infected_files),len(healthy_files)


#extracting path of images from diffrent directories and placing then into a data frame
import numpy as np
import pandas as pd

np.random.seed(42)

files_df = pd.DataFrame({
    'filename': infected_files + healthy_files,
    'label': ['malaria'] * len(infected_files) + ['healthy'] * len(healthy_files)
}).sample(frac=1, random_state=42).reset_index(drop=True)

files_df.head()


#splitting the data into test,train and validation of ration 30:60:10 respectively
train_files, test_files, train_labels, test_labels = train_test_split(files_df['filename'].values,
                                                                      files_df['label'].values, 
                                                                      test_size=0.3)
train_files,val_files,train_labels,val_labels=train_test_split(train_files,train_labels,test_size=0.1)
print(train_files.shape,val_files.shape,test_files.shape)


#train_files
# check the size using multi threading 
#To determine the average size of picture we process the images and find the average pixel to be used
#Introduing parllel threads for faster processing

from concurrent import futures
import threading

#introduing a thread executer
ex=futures.ThreadPoolExecutor(max_workers=None)
data_inp=[(idx,img,len(train_files)) for idx,img in enumerate(train_files)]

#defining a function that reads the images parllely
import cv2
def get_img_shape_parllel(idx,img,total_imgs):
    if idx%5000==0 or idx==(total_imgs-1):
        print ('{}:working on img num:{}'.format(threading.current_thread().name,idx))
        
    return cv2.imread(img).shape

train_img_dims_map=ex.map(get_img_shape_parllel,
                         [record[0] for record in data_inp],
                         [record[1] for record in data_inp],
                         [record[2] for record in data_inp])

train_img_dims_map
train_img_dims=list(train_img_dims_map)
train_img_dims

np.max(train_img_dims,axis=0)
np.min(train_img_dims,axis=0)

# image reshapes and resizing
img_dims=(125,125)

#reshaping the size of the image to img_dims size
def img_preprocess(idx,img,total_imgs):
    if idx%5000==0 or idx==(total_imgs-1):
        print("{} working on img num:{}".format(threading.current_thread().name,idx))
    img=cv2.imread(img)
    img=cv2.resize(img,dsize=img_dims,interpolation=cv2.INTER_CUBIC)
    img=np.array(img,dtype=np.float32)
    return img

ex=futures.ThreadPoolExecutor(max_workers=None)
train_data_inp=[(idx,img,len(train_files)) for idx,img in enumerate(train_files)]
val_data_inp=[(idx,img,len(val_files)) for idx,img in enumerate(val_files)]
test_data_inp=[(idx,img,len(test_files)) for idx,img in enumerate(test_files)]

#load images and transform
train_data_map=ex.map(img_preprocess,
                     [record[0]for record in train_data_inp],
                     [record[1]for record in train_data_inp],
                     [record[2]for record in train_data_inp])


val_data_map=ex.map(img_preprocess,
                   [record[0]for record in val_data_inp],
                   [record[1]for record in val_data_inp],
                   [record[2]for record in val_data_inp] 
                   )

test_data_map=ex.map(img_preprocess,
                    [record[0]for record in test_data_inp],
                    [record[1]for record in test_data_inp],
                    [record[2]for record in test_data_inp])

train_data=np.array(list(train_data_map))

train_data.shape

val_data=np.array(list(val_data_map))

val_data.shape

test_data=np.array(list(test_data_map))
test_data.shape

#defining the batch size and grayscaling the images for faster and efficent processing
batch_size=64
number_classes=2
epochs=10
input_shape=(125,125,3)

train_imgs_scaled=train_data/255.0
val_imgs_scaled=val_data/255.0
test_imgs_scaled=test_data/255.0
train_imgs_scaled.shape

#encoding the label values with new numeric values of [0,1]
from sklearn.preprocessing import LabelEncoder
#encoding lbels to binary
label=LabelEncoder()
label.fit(train_labels)
train_labels_encoded=label.transform(train_labels)
val_labels_encoded=label.transform(val_labels)
#type(label)
print(train_labels,train_labels_encoded)

#constructing a neural netwok with input ,output and 5 hidden layers( 3 convolution layers and 2 dense layers )
import tensorflow as tf
import keras
#developing a model.
#input layer
inp=tf.keras.layers.Input(shape=input_shape)
#convolution layer
conv1=tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation="relu",padding="same")(inp)
pool1=tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)
#convolution layer
conv2=tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation="relu",padding="same")(pool1)
pool2=tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)
conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

flat=tf.keras.layers.Flatten()(pool3)

hidden1 = tf.keras.layers.Dense(512, activation='relu')(flat)
drop1 = tf.keras.layers.Dropout(rate=0.3)(hidden1)
hidden2 = tf.keras.layers.Dense(512, activation='relu')(drop1)
drop2 = tf.keras.layers.Dropout(rate=0.3)(hidden2)

out=tf.keras.layers.Dense(1,activation="sigmoid")(drop2)

#defining the optimizer,loss function and  metrics that need to be shown
model = tf.keras.Model(inputs=inp, outputs=out)
model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
model.summary()

#training the neural network using train dataset and validating with val dataset
model.fit(x=train_imgs_scaled, y=train_labels_encoded, 
                    batch_size=batch_size,
                    epochs=1, 
                    validation_data=(val_imgs_scaled, val_labels_encoded), 
                    verbose=1)

#predicting the 
pred=model.predict(test_imgs_scaled)


#check the prediction.
test_labels[3],pred[3]