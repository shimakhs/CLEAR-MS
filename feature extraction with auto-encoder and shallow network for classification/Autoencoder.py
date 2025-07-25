# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 09:02:15 2022

@author: Shima
"""

import pickle
import matplotlib.pyplot as plt
import matplotlib
import cv2
import numpy as np 
import glob
import skimage.io as io
from skimage.transform import resize
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
# from imblearn.over_sampling import SMOTE

# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.under_sampling import TomekLinks
# from imblearn.under_sampling import EditedNearestNeighbours

# from imblearn.over_sampling import SMOTENC

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
from tensorflow.keras.layers import Activation

from sklearn.neighbors import NearestNeighbors

import gc 
from sklearn import metrics

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,balanced_accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import cv2
import numpy as np 



df2 = pd.DataFrame(columns =["Method",'Accuracy', 'Specificity','Sensitivity','gmean','balanced_accuracy',"f1_score","recall_score","precision_score"])
df_i = pd.DataFrame(columns =["Method",'Accuracy', 'Specificity','Sensitivity','gmean','balanced_accuracy',"f1_score","recall_score","precision_score"])
df_p = pd.DataFrame(columns =["Method",'Accuracy', 'Specificity','Sensitivity','gmean','balanced_accuracy',"f1_score","recall_score","precision_score"])
df_ip = pd.DataFrame(columns =["Method",'Accuracy', 'Specificity','Sensitivity','gmean','balanced_accuracy',"f1_score","recall_score","precision_score"])

def prapredata():
    
    with open('ScanPosition.pkl', 'rb') as f:  # defines boundaries are for right eye or not. right: 1, left = 0
          sp = pickle.load(f)   
    
    with open('XLayersBoundaryMap.pkl', 'rb') as f:  # predicted boundaries
          X = pickle.load(f)
    
    with open('TLayersBoundaryMap.pkl', 'rb') as f:  #target boundaries
          T = pickle.load(f)
          
    with open('HMlabels.pkl', 'rb') as f:  # HC = 0 , MS = 1 labels
          Y = pickle.load(f)    
          
    for i in range(len(sp)):
       if sp[i]:
          X[i] = X[i][:,:,::-1,:]
          T[i] = T[i][:,:,::-1,:]          
          
          a= []
    b=[]
    for j in range(len(X)):
       a= []
       for i in range(3):
          z = X[j][0,:,:,i] - X[j][0,:,:,i+1]
          #z = cv2.resize(z,(510,19))
          z = cv2.resize(z,(256,60))
          #z = cv2.resize(z,(224,224))

          a.append(z)
       a = np.array(a)
  # sum_a = np.max(a,axis=0)

  # print(a.shape)
       b.append(a)
    b = np.array(b)
    print(a.shape)
    print(b.shape)
    Y = np.array(Y)        
    return b,Y    
          
          
def repeat_vector(args):
    layer_to_repeat = args[0]
    
    return keras.layers.RepeatVector(25)(layer_to_repeat)

#def encoder(ly,neurons1=100,neurons2=100,neurons3=20):
def encoder(ly,neurons1=100,neurons2=50,neurons3=25):

    #inputs = keras.Input(shape=(ly, 19, 510))
    inputs = keras.Input(shape=(ly, 60, 256))
    #inputs = keras.Input(shape=(ly, 224, 224))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(neurons1, activation='tanh',activity_regularizer=keras.regularizers.l2(10e-6))(x)
    x = tf.keras.layers.Dense(neurons2, activation='tanh',activity_regularizer=keras.regularizers.l2(10e-6))(x)
    x = tf.keras.layers.Dense(neurons3 ,activation='tanh',activity_regularizer=keras.regularizers.l2(10e-6))(x)
   
    return keras.Model(inputs, x)



def decoder(ly,neurons1=100,neurons2=50,neurons3=25):

    inputs = keras.Input(shape=(neurons3,))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(neurons2, activation='tanh',activity_regularizer=keras.regularizers.l2(10e-6))(x)
    x = tf.keras.layers.Dense(neurons1, activation='tanh',activity_regularizer=keras.regularizers.l2(10e-6))(x)
    # x = tf.keras.layers.Dense(19 * 510 * ly, activation='tanh',activity_regularizer=keras.regularizers.l2(10e-6))(x)
    # x = tf.keras.layers.Reshape([ly, 19, 510])(x)
    x = tf.keras.layers.Dense(60 * 256 * ly, activation='tanh',activity_regularizer=keras.regularizers.l2(10e-6))(x)
    x = tf.keras.layers.Reshape([ly, 60, 256])(x)
    # x = tf.keras.layers.Dense(224 * 224 * ly, activation='tanh',activity_regularizer=keras.regularizers.l2(10e-6))(x)
    # x = tf.keras.layers.Reshape([ly, 224, 224])(x)
    
    return keras.Model(inputs, x)

def autoencoder(ly,neurons1=1,neurons2=1,neurons3=1,load=False):

    # input data
    #inputs = keras.Input(shape=(ly, 19, 510))
    inputs = keras.Input(shape=(ly, 60, 256))
    #inputs = keras.Input(shape=(ly, 224, 224))

    encoder_model = encoder(ly)
    decoder_model = decoder(ly)
    
    latent = encoder_model(inputs)
    # decoder_input = keras.layers.Lambda(repeat_vector, output_shape=(25,)) ([latent, inputs])
    outputs = decoder_model(latent)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.mse, metrics=['mse', 'mae'])
    if load == True:
        model = tf.keras.models.load_model("./ae.h5")
    return model, encoder_model, decoder_model         


class data_augmentation():
    def __init__(self, data, lambda_=0.5):

        self.target_data = data
        self.lambda_ = lambda_
        
    def interpolation_augmentation(self, data):

        k_n = self.clf.kneighbors(np.expand_dims(data, 0))
        tmp_context = []
        for ii in k_n[1][0][1:]:
            tmp_context.append((self.target_data[ii] - data) * self.lambda_ + data)
        return tmp_context
    
    def extrapolation_augmentation(self, data):

        k_n = self.clf.kneighbors(np.expand_dims(data, 0))
        tmp_context = []
        for ii in k_n[1][0][1:]:
            tmp_context.append((self.target_data[ii] - data) * self.lambda_ + data)
        return tmp_context
    
    def hard_extrapolation_augmentation(self, data):

        return np.expand_dims(data + self.lambda_ * (data + self.center), 0)
    
    def gaussian_noise(self, data, k):

        tmp_context = (np.repeat(np.expand_dims(np.array(data), axis=0), k, axis=0) + self.lambda_ * np.random.normal(np.zeros(self.std.shape), self.std, (k, self.std.shape[0]))).tolist()
        return tmp_context
    
    def augmentation_init(self, da_type, k):
        '''
        '''
        self.da_type = da_type
        self.k = k
        if da_type == 'norm':
            self.std = np.std(self.target_data, axis=0)
        elif da_type == 'hard_extra':
            self.center = np.mean(self.target_data, axis=0)
        else:
            self.clf = NearestNeighbors(n_neighbors=k+1) # include self selection 
            self.clf.fit(self.target_data)
        
    


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
#scores = np.array(scores)
#cm = confusion_matrix(np.argmax(labels_test, axis=1), np.argmax(pred1, axis=1))
# cm = confusion_matrix(np.array(y_test),np.array(pred)) 
# np.set_printoptions(precision=15)
# plot_confusion_matrix(cm = cm, normalize = False, target_names = ["HC","MS"])
# # plt.grid('off')
# plt.legend(prop={'size': 27})
# plt.savefig('/content/cm_denseNet121.png', dpi = 200, facecolor='w', edgecolor='w', orientation='landscape', papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=None, frameon=None)
# plt.show()





# base = a.copy()
def plh(history):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 300), history.history["accuracy"], label="train_accuracy")
    plt.plot(np.arange(0, 300), history.history["val_accuracy"], label="val_accuracy",linestyle='dashed')

    # Accuracy = accuracy_score(labels_valid, predictions.round(), normalize = True)
    # plt.title("Training Loss and Accuracy with dense121"+" Accuracy: %.2f%%" % (Accuracy * 100.0))
    plt.title("Training Accuracy")

    # plt.ylim((.6, .74))
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")#Loss/Accuracy

    plt.legend(loc=0)
    plt.savefig("./compare.jpg",dpi=250)



# base = a.copy()
def plhh(history):
      
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 300), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 300), history.history["val_loss"], label="val_loss",linestyle='dashed')






    # Accuracy = accuracy_score(labels_valid, predictions.round(), normalize = True)
    # plt.title("Training Loss and Accuracy with dense121"+" Accuracy: %.2f%%" % (Accuracy * 100.0))
    plt.title("Training Loss")

    # plt.ylim((.6, .74))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")#Loss/Accuracy

    plt.legend(loc=0)
    plt.savefig("./compare.jpg",dpi=250)


#layers = [[0,8],[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[0,2],[2,4],[4,6],[6,8],[0,4],[4,8]]
layers = [[0,3]]

for ly in layers:
    
    b,ytrain = prapredata()
    b = b [:,ly[0]:ly[1]]
    
    
    model, encoder_model, decoder_model = autoencoder(b.shape[1])
    ch = tf.keras.callbacks.ModelCheckpoint('./ae.h5',monitor= "loss",save_weights_only=False,save_best_only=True,mode='min')
    callbacks = [ch]
    history = model.fit(b, b, epochs=80, verbose=1,callbacks=callbacks)
    model, encoder_model, decoder_model = autoencoder(b.shape[1],load=True)
    x_train = encoder_model.predict(b)
    
    
    del b ,model
    gc.collect()
    x_train =np.array(x_train)
    



    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    i = 0
    yt =[]
    yp =[]
    yt_p =[]
    yt_i = []
    yt_ip= []
    yp_p =[]
    yp_i =[]
    yp_ip =[]
    
    for j , (train_index, test_index) in enumerate(kf.split(x_train,ytrain)):
        print("******************************************************")
        print("************FOLD NUM", j)
        X_train, X_test = x_train[train_index,:], x_train[test_index,:]
        y_train, y_test = ytrain[train_index], ytrain[test_index]
        tf.random.set_seed(42)
        model_classification = tf.keras.Sequential()
        model_classification.add(tf.keras.layers.Dense(50, activation='tanh'))
        model_classification.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model_classification.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0034),metrics=["accuracy"])

        ch = tf.keras.callbacks.ModelCheckpoint('./model.h5',monitor= "val_accuracy",save_weights_only=False,save_best_only=True,mode='max')
        callbacks = [ch]


#         history = model_classification.fit(X_train, y_train, epochs=300,validation_data=(X_test, y_test), verbose=0,callbacks = callbacks)

        da = data_augmentation(X_train)
        da.augmentation_init(da_type='hard_extra',k=1)
        #da.augmentation_init(da_type='new',k=1)

        x_train1 =[]
        ytrain1 =[]
        for i in range(len(y_train)):
          if y_train[i]==0:
            g = 0
            xt = da.hard_extrapolation_augmentation(X_train[i])
            #xt = da.interpolation_augmentation(X_train[i])

            x_train1.append(xt[0])
            x_train1.append(X_train[i])
            ytrain1.append(y_train[i])
            ytrain1.append(y_train[i])
            g += 1
            if g >2:
                break 

          else:
            x_train1.append(X_train[i])
            ytrain1.append(y_train[i])

        x_train1 =np.array(x_train1)
        ytrain1 =np.array(ytrain1)


        history = model_classification.fit(x_train1, ytrain1, epochs=300,validation_data=(X_test, y_test), verbose=1,callbacks = callbacks)

        # os = SMOTE(random_state = 0)
        # os_smote_X,os_smote_Y = os.fit_sample(X_train,y_train)

        # os =  EditedNearestNeighbours(sampling_strategy="majority")
        # os_smote_X,os_smote_Y = os.fit_sample(X_train,y_train)

        # from imblearn.pipeline import Pipeline
        # over = SMOTE(sampling_strategy=0.7)
        # under = RandomUnderSampler(sampling_strategy=1)
        # steps = [('o', over), ('u', under)]
        # pipeline = Pipeline(steps=steps)
        # os_smote_X,os_smote_Y = pipeline.fit_resample(X_train,y_train)
        # history = model_classification.fit(os_smote_X, os_smote_Y, epochs=300,validation_data=(X_test, y_test), verbose=0,callbacks = callbacks)


        model_classification = tf.keras.models.load_model("./model.h5")

        plh(history)

        print("******************************************************")

        plhh(history)

        print("******************************************************")
        print("******************************************************")

        # xintra,xextra = intra_hardextra(X_test)
        pred = model_classification.predict(X_test)
#         pred_p = model_classification.predict(x_test_ip[:32])
#         pred_i = model_classification.predict(x_test_ip[32:])
       # pred_ip = model_classification.predict(x_test_ip)

        
        
        # pred1 = model_classification.predict(xintra)
        # pred2 = model_classification.predict(xextra)
        # pred = ave_voting(pred,pred1,pred2)

        pred[pred>.5] = 1
        pred[pred<.5] = 0
        
#         pred_p[pred_p>.5] = 1
#         pred_p[pred_p<.5] = 0
        
#         pred_i[pred_i>.5] = 1
#         pred_i[pred_i<.5] = 0

        # pred_ip[pred_ip>.5] = 1
        # pred_ip[pred_ip<.5] = 0
        
        from sklearn import metrics
        cm1 = metrics.confusion_matrix(np.array(y_test), np.array(pred))
        total1=sum(sum(cm1))
        Accuracy = (cm1[0,0]+cm1[1,1])/total1
        Specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
        Sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])

        gmean = np.sqrt(Sensitivity * Specificity)
        balanced_accuracy = balanced_accuracy_score(np.array(y_test), np.array(pred))

        print ("Accuracy: %.4f "%Accuracy)
        print ("Specificity: %.4f " %Specificity)
        print ("Sensitivity: %.4f "%Sensitivity)
        print ("gmean: %.4f"%gmean)
        print ("balanced_accuracy: %.4f"%balanced_accuracy)
        print("******************************************************")
        print("******************************************************")


        print(classification_report(np.array(y_test),np.array(pred),digits=4))
        print("******************************************************")
        print("******************************************************")

        cm = confusion_matrix(np.array(y_test),np.array(pred)) 
        np.set_printoptions(precision=15)
        plot_confusion_matrix(cm = cm, normalize = False, target_names = ["HC","MS"])
        plt.show()
        print("******************************************************")
        print("******************************************************")
        yt.append(np.array(y_test).flatten())
        yp.append(np.array(pred).flatten())

#         yt_p.append(np.array(y_ip[:32]).flatten())
#         yp_p.append(np.array(pred_p).flatten())
        
#         yt_i.append(np.array(y_ip[32:]).flatten())
#         yp_i.append(np.array(pred_i).flatten())
        

        
        
        
    yy =[]
    pp = []
    for i in range(len(yt)):
        for j in range(len(yt[i])):
            yy.append(yt[i][j])
            pp.append(yp[i][j])
    
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

    f1_score = metrics.f1_score(np.array(yy),np.array(pp),average="weighted")
    recall_score = metrics.recall_score(np.array(yy),np.array(pp),average="weighted")
    precision_score = metrics.precision_score(np.array(yy),np.array(pp),average="weighted")
    numb =len(df2)+1
    mh = "Train All data ["+str(ly[0])+":"+str(ly[1]) +"]layer instance base, with hard_extrapolation_augmentation"    
    df2.loc[numb]= [ mh,Accuracy, Specificity,Sensitivity,gmean,balanced_accuracy,f1_score,recall_score,precision_score]
    df2.to_csv("./result.csv")
    '''
    yy =[]
    pp = []
    for i in range(len(yt_p)):
        for j in range(len(yt_p[i])):
            yy.append(yt_p[i][j])
            pp.append(yp_p[i][j])
    
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

    f1_score = metrics.f1_score(np.array(yy),np.array(pp),average="weighted")
    recall_score = metrics.recall_score(np.array(yy),np.array(pp),average="weighted")
    precision_score = metrics.precision_score(np.array(yy),np.array(pp),average="weighted")
    numb =len(df_p)+1
    mh = "Train UKBB data and test on petz ["+str(ly[0])+":"+str(ly[1]) +"]layer instance base, without aug"
    df_p.loc[numb]= [ mh,Accuracy, Specificity,Sensitivity,gmean,balanced_accuracy,f1_score,recall_score,precision_score]
    df_p.to_csv("./df_p.csv")
    
    
    
    
    
    yy =[]
    pp = []
    for i in range(len(yt_i)):
        for j in range(len(yt_i[i])):
            yy.append(yt_i[i][j])
            pp.append(yp_i[i][j])
    
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

    f1_score = metrics.f1_score(np.array(yy),np.array(pp),average="weighted")
    recall_score = metrics.recall_score(np.array(yy),np.array(pp),average="weighted")
    precision_score = metrics.precision_score(np.array(yy),np.array(pp),average="weighted")
    numb =len(df_i)+1
    mh = "Train UKBB data and test on iran ["+str(ly[0])+":"+str(ly[1]) +"]layer instance base, without aug"
    df_i.loc[numb]= [ mh,Accuracy, Specificity,Sensitivity,gmean,balanced_accuracy,f1_score,recall_score,precision_score]
    df_i.to_csv("./df_i.csv")
    '''
    
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
    
    

    
# plt.figure()
# plt.imshow(cv2.resize(b[10,2,:,:],(512,512)))  
    
    
fpr, tpr, _ = metrics.roc_curve(y_test,  pred)
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print('roc_auc_score for CNN: ', metrics.roc_auc_score(y_test, pred))
    













