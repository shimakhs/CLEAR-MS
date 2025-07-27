import numpy as np
import cv2
from sporco import plot
from sklearn.model_selection import train_test_split
import keras
from matplotlib import pyplot as plt
import numpy as np
# %matplotlib inline
from keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LeakyReLU
from keras.layers import Input,Dense,Flatten,Conv2D,MaxPooling2D,UpSampling2D,Dropout,AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import  RMSprop,Adam,SGD,Adadelta,Adagrad,Adamax,Nadam,Ftrl
from keras import regularizers
from sklearn.model_selection import train_test_split
from keras.layers import LeakyReLU
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,balanced_accuracy_score
from sklearn.utils import class_weight
from keras.regularizers import l2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys, os
import pickle
import seaborn as sns
import pandas as pd
import imgaug as ia
import imgaug.augmenters as iaa
import math
np.random.seed(1234)

# Getting back the objects:
with open('ScanPosition' + '.pkl', 'rb') as f:
   sp = pickle.load(f)
with open('XLayersBoundaryMap' + '.pkl', 'rb') as f:
   X = pickle.load(f)
with open('HMlabels' + '.pkl', 'rb') as f:
   Y  = pickle.load(f)

### flip all boundaries to be like right eye
for i in range(len(sp)):
  if sp[i]:
    X[i] = X[i][:,:,::-1,:]

dim1=60
dim2=256

def plot_img(img,name):

    fig = plot.figure(figsize=(14, 14))
    plot.imview(img, title=name, fig=fig)
    fig.show()

    return

def preprocess_data(X,Y):
    img=np.array([])
    dataset=np.zeros((len(X),dim1,dim2,9))
    for j in range(len(X)):
        img0=X[j]
        img=img0.squeeze()
        img=cv2.resize(img,(dim2,dim1))
        num=img.shape[2]
        def_img=np.zeros((img.shape))

        for k in range(0,num-1):
            def_img[:,:,k]=img[:,:,k+1]-img[:,:,k]
            #plot_img(def_img[:,:,k], name='img'+str(j)+'_def_layer'+str(k)+'Y'+str(Y[j]))


        def_img[:,:,num-1]=img[:,:,num-1]-img[:,:,0]
        #plot_img(def_img[:,:,num-1], name='img'+str(j)+'_def_layer'+str(8))

        dataset[j,:,:,:]=def_img
    return dataset

def final_dataset(dataset,Y,num_chanel,ch):
    new_dataset=np.zeros((len(dataset),dataset[0].shape[0],dataset[0].shape[1],len(ch)))

    for k in range(len(dataset)):

        img=dataset[k,:,:,:]
        new_img=np.zeros((img.shape[0],img.shape[1],num_chanel))

        for j in range(len(ch)):
            #new_img[:,:,j]=(img[:,:,ch[j]]/np.max(img[:,:,ch[j]])).astype('float32')
            I=img[:,:,ch[j]]
            new_img[:,:,j]=((I-np.min(I))/(np.max(I)-np.min(I))).astype('float32')

        new_dataset[k,:,:,:]=new_img


    Y=np.array(Y).squeeze()
    print('new_dataset:',new_dataset.shape)

    return new_dataset ,Y

#ch=[0,1,2,3,4,5,6,7,8] # layer number
ch=[0,1,2]
num_chanel=len(ch) # number of chanel
dataset=preprocess_data(X,Y)
print('dataset.shape:',dataset.shape)
new_dataset ,Y=final_dataset(dataset,Y,num_chanel,ch)
print('Y shape:',Y.shape)

new_dataset2= np.zeros((116,60,768))
new_dataset2[:,:,0:256] = new_dataset[:,:,:,0]
new_dataset2[:,:,256:512] = new_dataset[:,:,:,1]
new_dataset2[:,:,512:768] = new_dataset[:,:,:,2]

print(new_dataset2.shape)
new_dataset3 = np.expand_dims(new_dataset2, axis=3)
print(new_dataset3.shape)
plt.figure()
plt.imshow(new_dataset3[25,:,:,0])

unique, counts = np.unique(Y, return_counts=True)
label=dict(zip(unique, counts))
print('label:',label)

def CNN_model(neurons1=128,neurons2=64,neurons3=32,neurons4=16):

   inputs = keras.Input(shape=(60,768,1))
   x=keras.layers.Conv2D(neurons1, (5,5), activation='relu' ,kernel_constraint=keras.constraints.UnitNorm(axis=0 ), padding='same')(inputs)
   x=keras.layers.Conv2D(neurons1, (5,5), activation='relu', kernel_constraint=keras.constraints.UnitNorm(axis=0 ), padding='same')(x)
   x=keras.layers.MaxPooling2D(pool_size=(2, 2),data_format='channels_last')(x)
   x=keras.layers.BatchNormalization()(x)

   x=keras.layers.Conv2D(neurons2,  (4,4), activation='relu', kernel_constraint=keras.constraints.UnitNorm(axis=0 ), padding='same')(x)
   x=keras.layers.Conv2D(neurons2,  (3,3), activation='relu', kernel_constraint=keras.constraints.UnitNorm(axis=0 ), padding='same')(x)
   x=keras.layers.MaxPooling2D(pool_size=(2, 2),data_format='channels_last')(x)
   x=keras.layers.BatchNormalization()(x)


   x=keras.layers.Conv2D(neurons3,  (3,3), activation='relu', kernel_constraint=keras.constraints.UnitNorm(axis=0 ), padding='same')(x)
   x=keras.layers.Conv2D(neurons3,  (3,3), activation='relu', kernel_constraint=keras.constraints.UnitNorm(axis=0 ), padding='same')(x)
   x=keras.layers.MaxPooling2D(pool_size=(2, 2),data_format='channels_last')(x)
   x=keras.layers.BatchNormalization()(x)

   x=keras.layers.Conv2D(neurons4,  (3,3), activation='relu', kernel_constraint=keras.constraints.UnitNorm(axis=0 ), padding='same')(x)
   x=keras.layers.Conv2D(neurons4,  (3,3), activation='relu', kernel_constraint=keras.constraints.UnitNorm(axis=0 ), padding='same')(x)
   x=keras.layers.MaxPooling2D(pool_size=(2, 2),data_format='channels_last')(x)
   x=keras.layers.BatchNormalization()(x)

   x=keras.layers.Dropout(rate=0.25)(x)
   x=keras.layers.Flatten()(x)
   x=keras.layers.Dense(256,  kernel_constraint=keras.constraints.UnitNorm(axis=0),activation='relu')(x)#256
   x=keras.layers.Dense(128,  kernel_constraint=keras.constraints.UnitNorm(axis=0),activation='relu')(x)#128
   x=keras.layers.Dense(64,  kernel_constraint=keras.constraints.UnitNorm(axis=0),activation='relu')(x)#64
   x=keras.layers.Dense(32,  kernel_constraint=keras.constraints.UnitNorm(axis=0),activation='relu')(x)#32
   x=keras.layers.Dense(16,  kernel_constraint=keras.constraints.UnitNorm(axis=0),activation='relu')(x)#16
   x=keras.layers.Dense(2, activation='softmax')(x)
   return Model(inputs,x)

def classify_model():
    inputs=keras.Input(shape=(60,768,1))

    conv_model=CNN_model()
    outputs=conv_model(inputs)
    full_model=Model(inputs=inputs,outputs=outputs)
    full_model.summary()

    # full_model.compile(loss="binary_crossentropy", optimizer=SGD(),metrics=['accuracy'])#90 acc and more
    full_model.compile(loss="binary_crossentropy", optimizer=Adam(),metrics=['accuracy'])#97
    # full_model.compile(loss="categorical_crossentropy", optimizer=Adam(),metrics=['accuracy'])#97


    return full_model

def plot_loss(classify,itr):
    loss = classify.history['loss']
    val_loss = classify.history['val_loss']
    epochs = range(itr)
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    loss = classify.history['accuracy']
    val_loss = classify.history['val_accuracy']
    epochs = range(itr)
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training acc')
    plt.plot(epochs, val_loss, 'b', label='Validation acc')
    plt.title('Training and validation acc')
    plt.legend()
    plt.show()

# Confusion Matrix
import itertools
def plot_confusion_matrix(cm,
                          target_names,
                          title='',
                          #cmap=plt.cm.gist_yarg,
                          cmap='Blues',
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    plt.style.use('seaborn-ticks')
    plt.figure(figsize=(9,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=0)
    plt.yticks(tick_marks, target_names)




    fmt = '.3f' if normalize else 'd'


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize and cm[i, j] > 0:
            plt.text(j, i, format(cm[i, j],fmt),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        elif normalize == False and cm[i, j] > 0:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True labels')
    plt.xlabel('\nPredicted labels')

import tensorflow as tf
num_fold=5
balanced_acc_arr=[]
acc_arr=[]
skf = StratifiedKFold(n_splits=num_fold,shuffle=True,random_state=52)
skf.get_n_splits(new_dataset3, Y)
yt=[]
yp=[]
fold_counter=0

for train_index, test_index in skf.split(new_dataset3, Y):
      print('train_index:',len(train_index))
      print('test_ix:', len(test_index))
      fold_counter=fold_counter+1
      print('*****************fold_counter*********************',fold_counter)
      train_index=(np.array(train_index)).astype('uint8')
      test_index=(np.array(test_index)).astype('uint8')

      x_train  ,  X_test = new_dataset3[train_index] , new_dataset3[test_index]
      y_train, y_test = Y[train_index] , Y[test_index]

      # define and call model for classify images :
      model=classify_model()

      ch = keras.callbacks.ModelCheckpoint( '/content/drive/MyDrive/Colab Notebooks/last codes 18 bahman 1401/model_conv_concat.h5',monitor= "val_accuracy",save_weights_only=False,save_best_only=True,mode='max')
      callbacks = [ch]


      # agument images:

      from tensorflow.keras.utils import to_categorical
      #one_hot_train_label = to_categorical(new_y_train)

      one_hot_train_label = to_categorical(y_train)
      one_hot_test_label=to_categorical(y_test)

      #agmunt data:
      from keras.preprocessing.image import ImageDataGenerator
      datagen = ImageDataGenerator(
          horizontal_flip=True,
          vertical_flip=True,
          rotation_range=20,
          data_format='channels_last',
          fill_mode='nearest')

      datagen.fit(x_train)
      train_iterator = datagen.flow(x_train,one_hot_train_label ,batch_size=12,shuffle=True)#12
      print('Batches train=%d' % (len(train_iterator)))

      batch_x,batch_y = next(train_iterator)
      print('batch_x:',batch_x.shape)
      print('batch_y:',batch_y.shape)

      itr_epochs=300
      classify = model.fit( train_iterator ,epochs=itr_epochs,verbose=1,shuffle=True,validation_data=(X_test,one_hot_test_label),batch_size=12,callbacks=callbacks)

      plot_loss(classify,itr_epochs)

      # evalute model :
      model_classification = keras.models.load_model( '/content/drive/MyDrive/Colab Notebooks/last codes 18 bahman 1401/model_conv_concat.h5')
      y_pred = np.argmax(model_classification.predict(X_test), axis = 1)
      print('classes prediction:',y_pred)

      cm1 = metrics.confusion_matrix(np.array(y_test), np.array(y_pred))
      total1=sum(sum(cm1))
      Accuracy = (cm1[0,0]+cm1[1,1])/total1
      Specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
      Sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])

      acc_arr.append([Accuracy])

      gmean = np.sqrt(Sensitivity * Specificity)
      balanced_accuracy = balanced_accuracy_score(np.array(y_test), np.array(y_pred))
      balanced_acc_arr.append([balanced_accuracy])

      print ("Accuracy: %.4f "%Accuracy)
      print ("Specificity: %.4f " %Specificity)
      print ("Sensitivity: %.4f "%Sensitivity)
      print ("gmean: %.4f"%gmean)
      print ("balanced_accuracy: %.4f"%balanced_accuracy)
      print("******************************************************")
      print("******************************************************")


      print(classification_report(np.array(y_test),np.array(y_pred),digits=4))
      print("******************************************************")
      print("******************************************************")

      cm = confusion_matrix(np.array(y_test),np.array(y_pred))
      np.set_printoptions(precision=15)
      plot_confusion_matrix(cm = cm, normalize = False, target_names = ["HC","MS"])
      plt.show()
      print("******************************************************")
      print("******************************************************")
      yt.append(np.array(y_test).flatten())
      yp.append(np.array(y_pred).flatten())

print('avg_acc:',np.mean(np.array(acc_arr)))
print('avg_bacc:',np.mean(np.array(balanced_acc_arr)))

yy =[]
pp = []
for i in range(len(yt)):
  for j in range(len(yt[i])):
    yy.append(yt[i][j])
    pp.append(yp[i][j])

cm = confusion_matrix(np.array(yy),np.array(pp))
np.set_printoptions(precision=15)
plot_confusion_matrix(cm = cm, normalize = True, target_names = ["HC","MS"])
# plt.grid('off')
plt.legend(prop={'size': 27})
#plt.savefig('/content/cm_denseNet121.png', dpi = 200, facecolor='w', edgecolor='w', orientation='landscape', papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=None, frameon=None)
plt.show()

import matplotlib.cm as cm
from sklearn import metrics
cm1 = metrics.confusion_matrix(np.array(yy), np.array(pp))
total1=sum(sum(cm1))
Accuracy = (cm1[0,0]+cm1[1,1])/total1
Specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
Sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])

gmean = np.sqrt(Sensitivity * Specificity)
balanced_accuracy = balanced_accuracy_score(np.array(yy), np.array(pp))

print ("Accuracy: %.4f "%Accuracy)
print ("Specificity: %.4f " %Specificity)
print ("Sensitivity: %.4f "%Sensitivity)
print ("gmean: %.4f"%gmean)
print ("balanced_accuracy: %.4f"%balanced_accuracy)

print(classification_report(np.array(yy),np.array(pp),digits=4))

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)

plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print('roc_auc_score for CNN: ', metrics.roc_auc_score(y_test, y_pred))

model.summary()

neurons1=128
neurons2=64
neurons3=32
neurons4=16
model = tf.keras.Sequential()
model.add(keras.layers.Conv2D(neurons1, (5,5), activation='relu',input_shape=(60,768,1) ,kernel_constraint=keras.constraints.UnitNorm(axis=0), padding='same'))
model.add(keras.layers.Conv2D(neurons1, (5,5), activation='relu' ,kernel_constraint=keras.constraints.UnitNorm(axis=0), padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),data_format='channels_last'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(neurons2,  (4,4), activation='relu', kernel_constraint=keras.constraints.UnitNorm(axis=0 ), padding='same'))
model.add(keras.layers.Conv2D(neurons2,  (3,3), activation='relu', kernel_constraint=keras.constraints.UnitNorm(axis=0 ), padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),data_format='channels_last'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(neurons3,  (3,3), activation='relu', kernel_constraint=keras.constraints.UnitNorm(axis=0 ), padding='same'))
model.add(keras.layers.Conv2D(neurons3,  (3,3), activation='relu', kernel_constraint=keras.constraints.UnitNorm(axis=0 ), padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),data_format='channels_last'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(neurons4,  (3,3), activation='relu', kernel_constraint=keras.constraints.UnitNorm(axis=0 ), padding='same'))
model.add(keras.layers.Conv2D(neurons4,  (3,3), activation='relu', kernel_constraint=keras.constraints.UnitNorm(axis=0 ), padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),data_format='channels_last'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(rate=0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256,  kernel_constraint=keras.constraints.UnitNorm(axis=0),activation='relu'))
model.add(keras.layers.Dense(64,  kernel_constraint=keras.constraints.UnitNorm(axis=0),activation='relu'))
model.add(keras.layers.Dense(32,  kernel_constraint=keras.constraints.UnitNorm(axis=0),activation='relu'))
model.add(keras.layers.Dense(16,  kernel_constraint=keras.constraints.UnitNorm(axis=0),activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(loss="binary_crossentropy", optimizer=Adam(),metrics=['accuracy'])

model.summary()

import tensorflow as tf
num_fold=5
balanced_acc_arr=[]
acc_arr=[]
skf = StratifiedKFold(n_splits=num_fold,shuffle=True,random_state=52)
skf.get_n_splits(new_dataset3, Y)
yt=[]
yp=[]
fold_counter=0

for train_index, test_index in skf.split(new_dataset3, Y):
      print('train_index:',len(train_index))
      print('test_ix:', len(test_index))
      fold_counter=fold_counter+1
      print('*****************fold_counter*********************',fold_counter)
      train_index=(np.array(train_index)).astype('uint8')
      test_index=(np.array(test_index)).astype('uint8')

      x_train  ,  X_test = new_dataset3[train_index] , new_dataset3[test_index]
      y_train, y_test = Y[train_index] , Y[test_index]

      # define and call model for classify images :


      ch = keras.callbacks.ModelCheckpoint( '/content/drive/MyDrive/dr kafieh/model_conv.h5',monitor= "val_accuracy",save_weights_only=False,save_best_only=True,mode='max')
      callbacks = [ch]


      # agument images:

      from tensorflow.keras.utils import to_categorical
      #one_hot_train_label = to_categorical(new_y_train)

      one_hot_train_label = to_categorical(y_train)
      one_hot_test_label=to_categorical(y_test)

      #agmunt data:
      from keras.preprocessing.image import ImageDataGenerator
      datagen = ImageDataGenerator(
          horizontal_flip=True,
          vertical_flip=True,
          rotation_range=20,
          data_format='channels_last',
          fill_mode='nearest')

      datagen.fit(x_train)
      train_iterator = datagen.flow(x_train,one_hot_train_label ,batch_size=12,shuffle=True)#12
      print('Batches train=%d' % (len(train_iterator)))

      batch_x,batch_y = next(train_iterator)
      print('batch_x:',batch_x.shape)
      print('batch_y:',batch_y.shape)

      itr_epochs=300
      classify = model.fit( train_iterator ,epochs=itr_epochs,verbose=1,shuffle=True,validation_data=(X_test,one_hot_test_label),batch_size=12,callbacks=callbacks)

      plot_loss(classify,itr_epochs)

      # evalute model :
      model_classification = keras.models.load_model( '/content/drive/MyDrive/dr kafieh/model_conv.h5')
      y_pred = np.argmax(model_classification.predict(X_test), axis = 1)
      print('classes prediction:',y_pred)

      cm1 = metrics.confusion_matrix(np.array(y_test), np.array(y_pred))
      total1=sum(sum(cm1))
      Accuracy = (cm1[0,0]+cm1[1,1])/total1
      Specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
      Sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])

      acc_arr.append([Accuracy])

      gmean = np.sqrt(Sensitivity * Specificity)
      balanced_accuracy = balanced_accuracy_score(np.array(y_test), np.array(y_pred))
      balanced_acc_arr.append([balanced_accuracy])

      print ("Accuracy: %.4f "%Accuracy)
      print ("Specificity: %.4f " %Specificity)
      print ("Sensitivity: %.4f "%Sensitivity)
      print ("gmean: %.4f"%gmean)
      print ("balanced_accuracy: %.4f"%balanced_accuracy)
      print("******************************************************")
      print("******************************************************")


      print(classification_report(np.array(y_test),np.array(y_pred),digits=4))
      print("******************************************************")
      print("******************************************************")

      cm = confusion_matrix(np.array(y_test),np.array(y_pred))
      np.set_printoptions(precision=15)
      plot_confusion_matrix(cm = cm, normalize = False, target_names = ["HC","MS"])
      plt.show()
      print("******************************************************")
      print("******************************************************")
      yt.append(np.array(y_test).flatten())
      yp.append(np.array(y_pred).flatten())

print('avg_acc:',np.mean(np.array(acc_arr)))
print('avg_bacc:',np.mean(np.array(balanced_acc_arr)))

"""# ***Grad-CAM***"""

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

last_conv_layer_name = "conv2d_47"

grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )


img = X_test[10]

from tensorflow.keras.preprocessing import image
img_array = image.img_to_array(img)
imgg = np.expand_dims(img_array,axis=0)

# Remove last layer's softmax
model.layers[-1].activation = None

# Print what the top predicted class is
preds = model.predict(imgg)
#print("Predicted:", decode_predictions(preds, top=1)[0])

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(imgg, model, last_conv_layer_name)

# Display heatmap
#plt.matshow(heatmap)
#plt.show()
#print(imgg.shape)
#print(heatmap.shape)

ax = sns.heatmap(heatmap)

img = X_test[10]
img = np.squeeze(img)
hh = cv2.resize(heatmap,(768, 60))
superimposed_img2 = hh* 0.05+ img
ax3 = sns.heatmap(superimposed_img2)
