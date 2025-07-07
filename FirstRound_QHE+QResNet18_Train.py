
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:47:13 2024

@author: Zhou Yu
"""

print('Data loading  -----------------------------')
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import os
#import tensorflow as tf
from PIL import Image
import numpy as np


from QHE_function import QHE


def load_image(path):
    img = Image.open(path)
    resized_img = img.resize((100, 100))
#    resized_img = np.array(resized_img)
    return resized_img

#load RAF-DB dataset
data_dir = 'RAF_DB/Train/'
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]

#Temp variations
codes_list = []
labels = []
batch = []


with tf.Session() as sess:
    for each in classes:
        print("Starting {} images".format(each))
        class_path = data_dir + each
        files = os.listdir(class_path)
        for ii, file in enumerate(files, 1):
        # 载入图片并放入batch数组中
            image = load_image(os.path.join(class_path, file))
            image = np.array(image)
            #QHE processing
            img=QHE(image)
            i=img[:,:,0]
            j=img[:,:,1]
            k=img[:,:,2]
            img=np.zeros([img.shape[0],img.shape[1],4])
            img[:,:,1]=i
            img[:,:,2]=j
            img[:,:,3]=k
            batch.append(img.reshape((100, 100, 4)))
            labels.append(each)  
        else:
            continue
        
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
lb.fit(labels)

labels_vecs = lb.transform(labels)
codes = None

from sklearn import model_selection
codes, null1, labels_vectors, null2= model_selection.train_test_split(batch, labels_vecs, test_size=0.01, random_state=1)

#Import RAF-DB
data_dir2 = 'RAF_DB/Test/'
contents2 = os.listdir(data_dir2)
classes2 = [each for each in contents2 if os.path.isdir(data_dir2 + each)]

# codes_list for features
codes_list2 = []
# labels
labels22 = []
# batch for images
batch2 = []

with tf.Session() as sess:
    for each in classes2:
        print("Starting {} images".format(each))
        class_path2 = data_dir2 + each
        files2 = os.listdir(class_path2)
        for ii, file2 in enumerate(files2, 1):
            # 载入图片并放入batch数组中
            image2 = load_image(os.path.join(class_path2, file2))
            image2 = np.array(image2)
            img2=QHE(image2)
            i2=img2[:,:,0]
            j2=img2[:,:,1]
            k2=img2[:,:,2]
            img2=np.zeros([img2.shape[0],img2.shape[1],4])
            img2[:,:,1]=i2
            img2[:,:,2]=j2
            img2[:,:,3]=k2
            batch2.append(img2.reshape((100, 100, 4)))            
            labels22.append(each) 
        else:
            continue
        
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
lb.fit(labels)

labels_vecs2 = lb.transform(labels22)
codes2 = None
#Features are divided into 10 groups
from sklearn import model_selection
codes2, null1, labels_vectors2, null2= model_selection.train_test_split(batch2, labels_vecs2, test_size=0.01, random_state=1)
        

import os
import argparse
from keras import callbacks
from keras import layers, models, optimizers


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train),(x_val,y_val) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/(RAFDB)log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/RAFDBweights-{epoch:02d}.h5', monitor='val_accuracy',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=tf.optimizers.Adam(lr=args.lr),
                  loss='categorical_crossentropy',
#                  loss_weights=[1., args.lam_recon],
                  metrics=['accuracy']
                  )
  
    #Training without data augmentation:
    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
              validation_data=(x_val, y_val), callbacks=[log, tb, checkpoint, lr_decay])
    
    model.save_weights(args.save_dir + '/QWTR(RAF-DB)_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/(RAFDB)log.csv', show=True)

    return model


from   QResNet18K  import *
import matplotlib.pyplot as plt
K.set_image_data_format('channels_last') 

# setting the hyper parameters
parser = argparse.ArgumentParser(description="QResNet18 on FER.")
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch_size', default=20, type=int)
parser.add_argument('--lr', default=0.001, type=float,
                    help="Initial learning rate")
parser.add_argument('--lr_decay', default=0.4, type=float,
                    help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
parser.add_argument('--lam_recon', default=0.392, type=float,
                    help="The coefficient for the loss of decoder")
parser.add_argument('--debug', action='store_true',
                    help="Save weights by TensorBoard")
parser.add_argument('--save_dir', default='./QKresult')
parser.add_argument('-t', '--testing', action='store_true',
                    help="Test the trained model on testing dataset")
#parser.add_argument('--digit', default=5, type=int,
#                    help="Digit to manipulate")
parser.add_argument('-w', '--weights', default=None,
                    help="The path of the saved weights. Should be specified when testing")
args = parser.parse_args()
print(args)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

accuracy =[]

X_1 = []
y_1 = []

X_1 = np.array_split(codes,1,axis=0)
y_1 = np.array_split(labels_vectors,1,axis=0)

x_train =np.vstack(X_1[:]) 
y_train = np.vstack(y_1[:])

X_2 = []
y_2 = []

X_2 = np.array_split(codes2,1,axis=0)
y_2 = np.array_split(labels_vectors2,1,axis=0)

x_val =np.vstack(X_2[:]) 
y_val = np.vstack(y_2[:])

print('Train size : '+str(x_train.shape[0]))
print('Test size  : '+str(x_val.shape[0]))
#Load pretrain-model
#model.load_weights('QKresult/QQResNet18(RAF-DB)_model.h5')
model = qResNet18(input_shape=x_train.shape[1:])
#model.load_weights('QKresult/QWTR(RAF-DB)_model.h5')
#print(' ')
#print('Model Summary ----------------------------')

train(model=model, data=((x_train,y_train),(x_val,y_val)), args=args)
loss, acc =model.evaluate(x_val,y_val)
print('Test Loss = '+str(loss)+' | Test accuracy = '+str(acc))
print("That's All Folks :p ")
accuracy.append(acc)












    