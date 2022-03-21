#Contains the present best fitting emulator model for the 5PC LV Data
import numpy as np
import tensorflow as tf
import pickle
import scipy.stats as st
#loading in the pc data

sav_dirpc='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/cartesian_5pc_emulation_data'
m=np.genfromtxt(sav_dirpc+'/mean.txt')
mm=np.genfromtxt(sav_dirpc+'/mean_mesh.txt')
s=np.genfromtxt(sav_dirpc+'/sigma.txt')
pcs=np.genfromtxt(sav_dirpc+'/5comps.txt')
print(pcs.shape)
geom=np.genfromtxt('/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/test_data/geometry_data/new_tests.txt',delimiter=',')

mm=mm.reshape((1,17376))

proj=np.matmul((geom-m)/s-(mm-m)/s,pcs.T)



npc=5
y_train=np.genfromtxt('/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/cartesian_5pc_emulation_data/16000_trains/training_outs.txt',delimiter=',')
x_train=np.genfromtxt('/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/cartesian_5pc_emulation_data/16000_trains/training_ins.txt',delimiter=',')

def load_data(sav_dir,ext):
    for g in range(6):
        if g==0:
            tst_y=np.genfromtxt(sav_dir+'/outputs_geom'+str(g+1)+ext+'.txt',delimiter=',')
            tst_y=tst_y[range(5),:]
        else:
            ty=np.genfromtxt(sav_dir+'/outputs_geom'+str(g+1)+ext+'.txt',delimiter=',')
            tst_y=np.vstack([tst_y,ty[range(5),:]])
    for g in range(6):
        if g==0:
            tst_x=np.genfromtxt(sav_dir+'/inputs_geom'+str(g+1)+ext+'.txt',delimiter=',')
            tst_x=tst_x[range(5),:]
        else:
            tx=np.genfromtxt(sav_dir+'/inputs_geom'+str(g+1)+ext+'.txt',delimiter=',')
            tst_x=np.vstack([tst_x,tx[range(5),:]])
    tst_x=np.concatenate([np.ones((tst_x.shape[0],1)),tst_x,np.ones((tst_x.shape[0],1))],axis=1)
    return tst_x,tst_y

def load_datasyn(sav_dir,ext):
    for g in range(6,11):
        if g==6:
            tst_y=np.genfromtxt(sav_dir+'/outputs_geom'+str(g+1)+ext+'.txt',delimiter=',')
            #tst_y=np.zeros((5,25),dtype=np.float64)
            tst_y=tst_y[range(5),:]
        else:
            ty=np.genfromtxt(sav_dir+'/outputs_geom'+str(g+1)+ext+'.txt',delimiter=',')
            tst_y=np.vstack([tst_y,ty[range(5),:]])
    for g in range(6,11):
        if g==6:
            tst_x=np.genfromtxt(sav_dir+'/inputs_geom'+str(g+1)+ext+'.txt',delimiter=',')
            # tst_x=np.zeros((5,2),dtype=np.float64)
            tst_x=tst_x[range(5),:]
        else:
            tx=np.genfromtxt(sav_dir+'/inputs_geom'+str(g+1)+ext+'.txt',delimiter=',')
            tst_x=np.vstack([tst_x,tx[range(5),:]])
    tst_x=np.concatenate([np.ones((tst_x.shape[0],1)),tst_x,np.ones((tst_x.shape[0],1))],axis=1)
    return tst_x,tst_y

my=np.mean(y_train,axis=0)
sy=np.std(y_train,axis=0)
mx=np.mean(x_train,axis=0)
sx=np.std(x_train,axis=0)

y_train-=my
y_train/=sy
x_train-=mx
x_train/=sx

#directory to current best NN parameters
with open("/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/david_comparison/NN/training/strainsvolume/multout/tanh_160batches_lr-0_0005nhid1100nhid2100/network_pars.pickle", "rb") as input_file:
    pars = pickle.load(input_file)

wts=pars[0]
bis=pars[1]

weights = {
     'h1S': tf.Variable(wts[0],dtype=tf.float64,trainable=False),
     'h2S': tf.Variable(wts[1],dtype=tf.float64,trainable=False),
     'outS': tf.Variable(wts[2],dtype=tf.float64,trainable=False),
     'h21S': tf.Variable(wts[3],dtype=tf.float64,trainable=False),
     'out1S': tf.Variable(wts[4],dtype=tf.float64,trainable=False),
     'hid1outS': tf.Variable(wts[5],dtype=tf.float64,trainable=False),
 }

biases = {
    'b1S': tf.Variable(bis[0],dtype=tf.float64,trainable=False),
    'b2S': tf.Variable(bis[1],dtype=tf.float64,trainable=False),
    'outS': tf.Variable(bis[2],dtype=tf.float64,trainable=False),
}

def model(x,geom,mx=mx,sx=sx,nchain=1,batch=False): #The emulator model. This can be any model written in tensorflow
    if batch:
        X=tf.concat([tf.gather(x,[0,1,2,3],axis=1),geom],axis=1)
    else:
        X=tf.concat((tf.reshape(tf.gather(x,[0,1,2,3],axis=1),[nchain,4]),tf.tile(tf.reshape(geom,[1,5]),[nchain,1])),axis=1)
    X=tf.divide(tf.subtract(X,mx),sx)
    h1S=tf.nn.tanh(tf.matmul(X,weights['h1S'])+biases['b1S'])
    h2S=tf.nn.tanh(tf.matmul(h1S,weights['h2S'])+tf.matmul(X,weights['h21S'])+biases['b2S'])
    outS=tf.matmul(h2S,weights['outS'])+tf.matmul(X,weights['out1S'])+tf.matmul(h1S,weights['hid1outS'])+biases['outS']
    return outS

# with open("/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/david_comparison/NN/training/strainsvolume/multout/tanh_160batches_lr-0_0005nhid1100nhid2100/network_pars.pickle", "rb") as input_file:
#     pars = pickle.load(input_file)
