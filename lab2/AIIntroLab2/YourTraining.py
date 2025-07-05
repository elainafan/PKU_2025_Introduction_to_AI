import mnist
import numpy as np
import pickle
import time
from util import setseed
from autograd.utils import PermIterator
from autograd.BaseGraph import Graph
from autograd.BaseNode import *
from scipy.ndimage import rotate,shift

setseed(0) 
begin_time=time.time()

save_path="model/My_train_MLPmodel1.npy"
X=np.concatenate([mnist.trn_X,mnist.val_X],axis=0)
Y=np.concatenate([mnist.trn_Y,mnist.val_Y],axis=0)
lr=2e-3
wd1=0
wd2=2e-5
batchsize=128

def augment_data(dataset,label):
    augmented_data=[]
    augmented_label=[]
    for i,img in enumerate(dataset):
        img=img.reshape(28,28)
        rotated=rotate(img,np.random.uniform(-15,15),reshape=False)
        augmented_data.append(rotated.reshape(-1))
        augmented_label.append(label[i])
        translated=shift(img,[np.random.randint(-5,5),np.random.randint(-5,5)])
        augmented_data.append(translated.reshape(-1))
        augmented_label.append(label[i])
    return (np.array(augmented_data),np.array(augmented_label))

temp_X,temp_Y=augment_data(X,Y)
X=np.concatenate([X,temp_X],axis=0)
Y=np.concatenate([Y,temp_Y],axis=0)

if __name__=="__main__":
    graph=Graph([Linear(784,256),BatchNorm(256),relu(),Linear(256,128),
                 BatchNorm(128),relu(),Linear(128,64),BatchNorm(64),relu(),
                 Linear(64,32),BatchNorm(32),relu(),
                 Linear(32,10),LogSoftmax(),NLLLoss(Y)])
    best_acc=0
    dataloader=PermIterator(X.shape[0],batchsize)
    temp_time0=time.time()
    for i in range(1,30+1):
        thaty=[]
        ty=[]
        tloss=[]
        graph.train()
        for perm in dataloader:
            tem_X=X[perm]
            tem_Y=Y[perm]
            graph[-1].y=tem_Y
            graph.flush()
            pred,loss=graph.forward(tem_X)[-2:]
            thaty.append(np.argmax(pred,axis=1))
            ty.append(tem_Y)
            graph.backward()
            graph.optimstep(lr,wd1,wd2)
            tloss.append(loss)
        loss=np.average(tloss)
        acc = np.average(np.concatenate(thaty)==np.concatenate(ty))
        print(f"epoch{i}loss{loss:3e}acc{acc:4f}")
        if acc>best_acc:
            best_acc=acc
            with open(save_path,"wb") as f:
                pickle.dump(graph,f)
    temp_time1=time.time()
    print("tr_time",temp_time1-temp_time0)
    finish_time=time.time()
    print("total_time",finish_time-begin_time)
