import math
from builtins import enumerate
import keras
from sklearn.metrics import confusion_matrix,roc_auc_score
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Lambda
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import save_model, load_model
import random
from imblearn.keras import BalancedBatchGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from sklearn.metrics import confusion_matrix
from tensorboard.plugins.hparams import api as hp
from imblearn.keras import BalancedBatchGenerator, balanced_batch_generator
from scipy.ndimage.filters import median_filter
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from tensorflow.keras.layers import concatenate, Dense, Dropout, Input, Activation, Flatten, Conv2D, MaxPooling2D, \
    AveragePooling2D, GlobalAveragePooling2D, BatchNormalization
from modules.Sampler import Sampler
from modules import LoadData as ld
from modules.LoadData import load_data
from modules.Generators import *
from modules.CallBacks import *
import os
import pickle

class DTL():
    @staticmethod
    def load():
        from PIL import Image
        import os
        x_val=[]
        x_train=[]
        x_test=[]
        y_val=[]
        y_train=[]
        y_test=[]
        def crop_center(img,cropx,cropy):
          y,x,_ = img.shape
          startx = x//2-(cropx//2)
          starty = y//2-(cropy//2)
          return np.array(img[starty:starty+cropy,startx:startx+cropx,:])
        li=os.listdir("/content/HuSHem/01_Normal")
      
        for l in range(len(li)):
          img=Image.open("/content/HuSHem/01_Normal/"+li[l])
          img=np.array(img)
          img=crop_center(img,70,70)
          x_train.append(img)
          y_train.append(0)

          
        li=os.listdir("/content/HuSHem/02_Tapered")
        i=int(0.8*len(li))
        j=int(0.2*len(li))
        for l in range(len(li)):
          img=Image.open("/content/HuSHem/02_Tapered/"+li[l])
          img=np.array(img)
          img=crop_center(img,70,70)
          x_train.append(img)
          y_train.append(1)
  

        li=os.listdir("/content/HuSHem/03_Pyriform")

        for l in range(len(li)):
          img=Image.open("/content/HuSHem/03_Pyriform/"+li[l])
          img=np.array(img)
          img=crop_center(img,70,70)
          x_train.append(img)
          y_train.append(2)


        li=os.listdir("/content/HuSHem/04_Amorphous")
        i=int(0.8*len(li))
        j=int(0.2*len(li))
        for l in range(len(li)):
          img=Image.open("/content/HuSHem/04_Amorphous/"+li[l])
          img=np.array(img)
          img=crop_center(img,70,70)

          x_train.append(img)
          y_train.append(3)
        
        
        li=os.listdir("/content/fake/no")
        i=int(0.7*len(li))
        j=int(0.3*len(li))
        for l in range(len(li)):
          img=Image.open("/content/fake/no/"+li[l])
          img=np.array(img)
          img=crop_center(img,70,70)
          x_train.append(img)
          y_train.append(0)



        li=os.listdir("/content/fake/ta")
        i=int(0.7*len(li))
        j=int(0.3*len(li))
        for l in range(len(li)):
          img=Image.open("/content/fake/ta/"+li[l])
          img=np.array(img)
          img=crop_center(img,70,70)
          x_train.append(img)
          y_train.append(1)



        li=os.listdir("/content/fake/py")
        i=int(0.7*len(li))
        j=int(0.3*len(li))
        for l in range(len(li)):
          img=Image.open("/content/fake/py/"+li[l])
          img=np.array(img)
          img=crop_center(img,70,70)
          x_train.append(img)
          y_train.append(2)
   


        li=os.listdir("/content/fake/ap")
        
        for l in range(len(li)):
          img=Image.open("/content/fake/ap/"+li[l])
          img=np.array(img)
          img=crop_center(img,70,70)
          x_train.append(img)
          y_train.append(3)
        y_train=np.array(y_train)
        x_train=np.array(x_train)
        tmp_idx = np.arange(x_train.shape[0])
        np.random.shuffle(tmp_idx)
        x_train=x_train[tmp_idx]
        y_train=y_train[tmp_idx]
        x=x_train.copy()
        y=y_train.copy()
        
        i=int(0.6*len(x))
        j=int(0.2*len(x))
        x_train=x[:i]
        y_train=y[:i]
        
        x_val=x[i:i+j]
        y_val=y[i:i+j]
        
        y_test=y[i+j:]
        x_test=x[i+j:]
 


        ar=np.unique(y_train,return_counts=True)[1]
        maxi=np.max(np.unique(y_train,return_counts=True)[1])
        i_maxi=np.argmax(np.unique(y_train,return_counts=True)[1])
        for i in range(4):
          if i==i_maxi:
            continue
         
          s=maxi-ar[i]
          for j in range(len(x_train)):
            if y_train[j]==i:
              x_train=np.append(x_train,[x_train[j]],axis=0)
              y_train=np.append(y_train,[i],axis=0)
              s-=1
              if s==0:
                break
        print(np.unique(y_train,return_counts=True)[1])

        ar=np.unique(y_test,return_counts=True)[1]
        maxi=np.max(np.unique(y_test,return_counts=True)[1])
        i_maxi=np.argmax(np.unique(y_test,return_counts=True)[1])
        for i in range(4):
          if i==i_maxi:
            continue
         
          s=maxi-ar[i]
          for j in range(len(x_test)):
            if y_test[j]==i:
              x_test=np.append(x_test,[x_test[j]],axis=0)
              y_test=np.append(y_test,[i],axis=0)
              s-=1
              if s==0:
                break
        print(np.unique(y_test,return_counts=True)[1])


        y_test=np.array(y_test)
        y_val=np.array(y_val)
        y_train=np.array(y_train)
        y_test = y_test.reshape(len(y_test), 1)
        y_val = y_val.reshape(len(y_val), 1)
        y_train = y_train.reshape(len(y_train), 1)
        num_classes = 4
        y_test = (y_test== np.arange(num_classes).reshape(1, num_classes))*1
        y_val = (y_val==np.arange(num_classes).reshape(1, num_classes))*1
        y_train = (y_train==np.arange(num_classes).reshape(1, num_classes))*1
      
       
        data={}
        data["x"]=x_train
        data["y"]=y_train
        
        data["x_test"]=np.array(x_test)
        data["y_test"]=np.array(y_test)

        
        data["x_val"]=np.array(x_val)
        data["y_val"]=np.array(y_val)
        
        data["x_train"]=np.array(x_train)
        data["y_train"]=np.array(y_train)
        import pickle
        with open('data' + '.pkl', 'wb') as f:
          pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        with open('data' + '.pkl', 'rb') as f:
          data=pickle.load(f)
        
        return data
    @staticmethod
    def load2():
        from PIL import Image
        import os
        x_val=[]
        x_train=[]
        x_test=[]
        y_val=[]
        y_train=[]
        y_test=[]
        def crop_center(img,cropx,cropy):
          y,x,_ = img.shape
          startx = x//2-(cropx//2)
          starty = y//2-(cropy//2)
          return np.array(img[starty:starty+cropy,startx:startx+cropx,:])
        li=os.listdir("/content/HuSHem/01_Normal")
        i=int(0.8*len(li))
        j=int(0.2*len(li))
        for l in range(len(li)):
          img=Image.open("/content/HuSHem/01_Normal/"+li[l])
          img=np.array(img)
          img=crop_center(img,70,70)
          x_train.append(img)
          y_train.append(0)

          
        li=os.listdir("/content/HuSHem/02_Tapered")
        i=int(0.8*len(li))
        j=int(0.2*len(li))
        for l in range(len(li)):
          img=Image.open("/content/HuSHem/02_Tapered/"+li[l])
          img=np.array(img)
          img=crop_center(img,70,70)
          x_train.append(img)
          y_train.append(1)
  

        li=os.listdir("/content/HuSHem/03_Pyriform")

        for l in range(len(li)):
          img=Image.open("/content/HuSHem/03_Pyriform/"+li[l])
          img=np.array(img)
          img=crop_center(img,70,70)
          x_train.append(img)
          y_train.append(2)


        li=os.listdir("/content/HuSHem/04_Amorphous")
        i=int(0.8*len(li))
        j=int(0.2*len(li))
        for l in range(len(li)):
          img=Image.open("/content/HuSHem/04_Amorphous/"+li[l])
          img=np.array(img)
          img=crop_center(img,70,70)

          x_train.append(img)
          y_train.append(3)
        
        
        li=os.listdir("/content/fake/no")
        i=int(0.7*len(li))
        j=int(0.3*len(li))
        for l in range(len(li)):
          img=Image.open("/content/fake/no/"+li[l])
          img=np.array(img)
          img=crop_center(img,70,70)
          x_train.append(img)
          y_train.append(0)



        li=os.listdir("/content/fake/ta")
        i=int(0.7*len(li))
        j=int(0.3*len(li))
        for l in range(len(li)):
          img=Image.open("/content/fake/ta/"+li[l])
          img=np.array(img)
          img=crop_center(img,70,70)
          x_train.append(img)
          y_train.append(1)



        li=os.listdir("/content/fake/py")
        i=int(0.7*len(li))
        j=int(0.3*len(li))
        for l in range(len(li)):
          img=Image.open("/content/fake/py/"+li[l])
          img=np.array(img)
          img=crop_center(img,70,70)
          x_train.append(img)
          y_train.append(2)
   


        li=os.listdir("/content/fake/ap")
        
        for l in range(len(li)):
          img=Image.open("/content/fake/ap/"+li[l])
          img=np.array(img)
          img=crop_center(img,70,70)
          x_train.append(img)
          y_train.append(3)
        y_train=np.array(y_train)
        x_train=np.array(x_train)
        
        

        y_train=np.array(y_train)
        y_train = y_train.reshape(len(y_train), 1)
        num_classes = 4
        y_train = (y_train==np.arange(num_classes).reshape(1, num_classes))*1
      
        data={}
        data["x"]=x_train
        data["y"]=y_train
        
        import pickle
        with open('data' + '.pkl', 'wb') as f:
          pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        with open('data' + '.pkl', 'rb') as f:
          data=pickle.load(f)
        
        return data
    def phase2(self,epochs, load_best_weigth, verbose, TensorB, name_of_best_weight,phase):
      for l in self.__model.layers[:-14]:
        l.trainable=True
      self.__model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.000001), metrics=["accuracy"], loss='categorical_crossentropy')
      self.train(epochs, load_best_weigth, verbose, TensorB, name_of_best_weight,phase)
    def __init__(self, params,base_model,label,data=None):
        default_params = {"agumentation": False, "scale": False, "dense_activation": "relu", "regularizition": 0.0
            , "dropout": 0.0, "optimizer": "adam", "number_of_dense": 1, "balancer": "None", "batch_size": 32}
        default_params.update(params)
        Model = base_model
        params = default_params
        # if data==None:
        #   data = load_data(label=label, phase="search")
        self.batch_size = params["batch_size"]
        # if params['agumentation']:
        #     data["x_val"] = ld.normalize(data["x_val"])
        import pandas as pd
        df=pd.read_csv("/content/1.csv")
        df.columns=['A','B']

        for i in range(2,23):
          s=pd.read_csv("/content/"+str(i)+".csv")
          s.columns=['A','B']
          df=df.append(s)
  
        from PIL import Image
        import numpy as np
        
        label={"Non-Viable-Tumor":[],"Non-Tumor":[],"Viable":[]}
        for i in range(len(df)):
          a=df.iloc[i]['A']
          tmp=a.replace(" - ","-")[:-3]
          tmp=tmp.replace(" ","-")
          try:
            img=Image.open("/content/resized/"+tmp+"JPG")
            img=np.array(img)
            label[df.iloc[i]['B']].append(img)
          except:
            continue
        for k in label.keys():
          tmp_idx = np.arange(len(label[k]))
          np.random.shuffle(tmp_idx)
          label[k] =list(np.array(label[k])[tmp_idx])
        x_train=[]
        y_train=[]
        x_val=[]
        y_val=[]
        x_test=[]
        y_test=[]
        num_classes = 3
        t=0
        for v in label.values():
          i=int(len(v)*0.7)
          j=int(len(v)*0.2)
          x_train+=v[:i]
          y_train+=[[t] for i in range(len(v[:i]))]
          x_test+=v[i:i+j]
          y_test+=[[t] for i in range(len(v[i:i+j]))]
          x_val+=v[i+j:]
          y_val+=[[t] for i in range(len(v[i+j:]))]
          t+=1
        
        y_train=np.array(y_train)
        y_val=np.array(y_val)
        y_test=np.array(y_test)
        x_train=np.array(x_train)
        x_val=np.array(x_val)
        x_test=np.array(x_test)

        y_train = (y_train==np.arange(num_classes).reshape(1, num_classes))*1
        y_test = (y_test==np.arange(num_classes).reshape(1, num_classes))*1
        y_val = (y_val==np.arange(num_classes).reshape(1, num_classes))*1
        
        print(y_val.shape,y_test.shape)
        data = ld.fix_data(False, x_train, y_train,x_val,y_val,x_test, y_test)


        #     data["x_test"] = ld.normalize(data["x_test"])
        # elif params["scale"]:
        #     data["x_val"] = ld.normalize(data["x_val"])
        #     data["x_test"] = ld.normalize(data["x_test"])
        #     data["x_train"] = ld.normalize(data["x_train"])

        data["x_test"]=(data["x_test"]-127.5)/127.5
        data["x_val"]=(data["x_val"]-127.5)/127.5
        data["x_train"]=(data["x_train"]-127.5)/127.5
        # data=DTL.load2()
        # print(len(data["y_test"]),len(data["y_val"]),len(data["y_train"]))
        regularization = not (params["regularizition"] == 0.0)

        dropout = not (params["dropout"] == 0.0)

        self.agumnetation = params["agumentation"]

        ############ Creating CNN ##############
        optimizer = params["optimizer"]
        # inp = Input((64,64, 1))
        # con = concatenate([inp, inp, inp])
        import keras
        # model = keras.models.load_model(address)
        model = Model(include_top=False, weights='imagenet', input_shape=(128,128,3))
        x = Flatten()(model.layers[-1].output)
        # for l in model.layers:
        #   l.trainable=False
        for i in range(params["number_of_dense"]):
            if regularization:
                x = Dense(params["nn"], activation=params["dense_activation"],
                          kernel_regularizer=l2(params["regularizition"]))(x)
            else:
                x = Dense(params["nn"], activation=params["dense_activation"])(x)
            if dropout:
                x = Dropout(params["dropout"])(x)
        x = Dense(3, activation="softmax", name="classification")(x)
        model = tf.keras.Model(model.input, x)
        model.compile(optimizer="Adadelta", metrics=["accuracy"], loss=params["loss"])
        # model.load_weights("w.h5")
        
        self.__model = model
        self.__data = data
        
        self.balancer = params["balancer"]
        self.__number_of_dense = params["number_of_dense"]
        self.details = [list(params.keys())[i] + ":" + str(list(params.values())[i]) for i in range(len(params))]

    def train(self, epochs, load_best_weigth, verbose, TensorB, name_of_best_weight, phase):
        if phase == "train" and self.agumnetation:
            self.__data["x_train"] = self.__data["x_train_128"]
        # self.__data["x_val"] = self.__data["x_val_128"]
        batch_size = self.batch_size
        balancer = self.balancer
        callbacks=[]
        callbacks = [DTL_ModelCheckpoint(self.__data["x_val"], self.__data["y_val"], self.__model, name_of_best_weight)]
        if TensorB:  # save History of train
            tb = TensorBoard(log_dir="log_train/" + "#".join(self.details))
            callbacks.append(tb)
        '''
        Here we have two kinds of sampler.
        online,offline
        online:for fix data on every epoch(we will use it in fit_generator)
        offline:for fix data before train

        onlines 1-batch_balancer:we can just use Data Agumentation with it
                2-DySa:nothing can't combine with it
        offlines:if balamcer be one of {smote,adasyn,None},then we should just our balancer to fix data and then use agumentation or not.

        '''
        acc=[]
        loss=[]
        ##batch_balancer
        if balancer == "batch_balancer":
            S = [[], []]
            for i in range(len(self.__data["x_train"])):
                S[self.__data["y_train"][i][0]].append(self.__data["x_train"][i])
            S = np.array(S)
            generator = BatchBalancer(S, self.agumnetation, batch_size)
            hist = self.__model.fit_generator(generator, validation_data=(self.__data["x_val"], self.__data["y_val"]),
                                              shuffle=True, callbacks=callbacks, steps_per_epoch=1000 / batch_size,
                                              epochs=epochs, verbose=verbose)
        else:  ###It means that we will use offline balancers
            self.__data = self.__getBalancedData(balancer)  # fixing data
            if self.agumnetation:
                generator = Agumentation(self.__data, batch_size)
                hist = self.__model.fit_generator(generator,
                                                  validation_data=(self.__data["x_val"], self.__data["y_val"]),
                                                  shuffle=True, steps_per_epoch=1000 / batch_size,
                                                  epochs=epochs, verbose=verbose)
            ###It means that we will use offline balancers
            else:
                self.__model.fit(self.__data["x_train"], self.__data["y_train"],
                                                batch_size=batch_size,epochs=epochs,validation_data=(self.__data["x_val"], self.__data["y_val"]),
                                               shuffle=True,callbacks=callbacks,
                                                 verbose=verbose)
                        
                        # loss.append(h.history['loss'][0])
                        # acc.append(h.history['accuracy'][0])
        if load_best_weigth:
            self.__model.load_weights(name_of_best_weight)
        
        save_model(self.__model, "model_" + name_of_best_weight)
        # np.save(name_of_best_weight+"_loss.npy",loss)
        # np.save(name_of_best_weight+"_acc.npy",acc)
        # cleaning model from GPU

    def clear(self):
        tf.keras.backend.clear_session()
        del self.__model
        del self.__data

    def __getBalancedData(self, name):
        if name == "None":  # no sampler
            return self.__data
        elif name == "smote":
            return Sampler("smote", 10, self.__data).run()
        elif name == "adasyn":
            return Sampler("adasyn", 10, self.__data).run()

    def evaluate(self):
        X = self.__data["x_test"]
        y = self.__data["y_test"]
        # y_pred1 = self.__model.predict(X)
        # print(y_pred1.argmax(axis=1).shape,y_pred1.shape)
        # acc=np.sum(y.argmax(axis=1)==y_pred1.argmax(axis=1))/47
        acc=self.__model.evaluate(X,y)
        # y_pred = y_pred1 > 0.5
        # y_pred = y_pred * 1
        # c_matrix = confusion_matrix(y, y_pred)
        # precision = c_matrix[0, 0] / sum(c_matrix[:,0])
        # recall = c_matrix[0, 0] / sum(c_matrix[ 0])
        # acc = np.sum(c_matrix.diagonal()) / np.sum(c_matrix)
        # f_half = 1.25 * precision * recall / (.25 * precision + recall)
        # g_mean = math.sqrt(precision * recall)
        # TP = c_matrix[0][0]
        # TN = c_matrix[1][1]
        # FN= c_matrix[0][1]
        # FP= c_matrix[1][0]
        # mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        # auc = 0
        return acc
    @staticmethod
    def k_fold(k,label, epochs, params, load_best_weigth, verbose, TensorB, name_of_best_weight,base_model):
        flag = params['agumentation']
        data =DTL.load2()
        results=[]
        size = len(data['x']) // k
        print(size)
        tmp_idx = np.arange(data['x'].shape[0])
        np.random.shuffle(tmp_idx)
        x = data['x'][tmp_idx]
        y = data['y'][tmp_idx]
        np.save("x.npy", x)
        np.save("y.npy", y)
        acc= []
        for i in range(k):
            x_test = x[i * size:(i + 1) * size]
            y_test = y[i * size:(i + 1) * size]
            x_train = np.append(x[0:i * size], x[(i + 1) * size:], axis=0)
            y_train = np.append(y[0:i * size], y[(i + 1) * size:], axis=0)
            tmp = random.sample(range(len(x_train)), int(len(x_train)*0.2))
            x_val = []
            y_val = []
            # for j in tmp:
            #     x_val.append(x_train[j])
            #     y_val.append(y_train[j])
            # x_val = np.array(x_val)
            # y_val = np.array(y_val)
            # x_train = np.delete(x_train, tmp, axis=0)
            # y_train = np.delete(y_train, tmp, axis=0)

            ##########fixing data#########
            data = ld.fix_data(flag, x_train, y_train,x_val,y_val,x_test, y_test)

            model = DTL(params=params,base_model=base_model,label=label,data=data)
            model.train(epochs, load_best_weigth, verbose, TensorB, name_of_best_weight + str(i) + ".h5", "k_fold")
            model.phase2(epochs, load_best_weigth, verbose, TensorB, "p2"+name_of_best_weight + str(i) + ".h5", "k_fold")
            results.append(model.evaluate())
            print(results[-1])
            acc.append(results[-1][1])
            model.clear()
            del model
        
        return results,np.mean(acc)





