import tensorflow as tf
import tensorflow_probability as tfp
tfd=tfp.distributions
import gpflow
import pickle
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
sav_dir='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/10pc_emulator/cartesian_10pc_emulation_data/'
ext=''


Y30=np.genfromtxt(sav_dir+'training_outsklotz30.txt',delimiter=',')
XV30=np.genfromtxt(sav_dir+'30_invols.txt',delimiter=',')
XP30=np.genfromtxt(sav_dir+'training_insklotz30.txt',delimiter=',')
XWV30=np.genfromtxt(sav_dir+'30_inWV.txt',delimiter=',')
X30=np.concatenate([XP30[:,range(4)],XV30.reshape((9000,1)),XWV30.reshape((9000,1))],axis=1)
Ytest30=np.genfromtxt(sav_dir+'test_outsklotz30.txt',delimiter=',')
XtestV30=np.genfromtxt(sav_dir+'30_involstest.txt',delimiter=',')
XtestP30=np.genfromtxt(sav_dir+'test_insklotz30.txt',delimiter=',')
# Xtest30=np.concatenate([XtestP30[:,range(4)],XtestV30.reshape((len(XtestV30),1))],axis=1)
XtestWV30=np.genfromtxt(sav_dir+'30_inWVtest.txt',delimiter=',')
Xtest30=np.concatenate([XtestP30[:,range(4)],XtestV30.reshape((len(XtestV30),1)),XtestWV30.reshape((len(XtestWV30),1))],axis=1)


Y20=np.genfromtxt(sav_dir+'training_outsklotz.txt',delimiter=',')
XV20=np.genfromtxt(sav_dir+'20_invols.txt',delimiter=',')
XP20=np.genfromtxt(sav_dir+'training_insklotz.txt',delimiter=',')
XWV20=np.genfromtxt(sav_dir+'20_inWV.txt',delimiter=',')
XtestWV20=np.genfromtxt(sav_dir+'20_inWVtest.txt',delimiter=',')
XP20=np.array(XP20[range(9000),:])
Y20=np.array(Y20[range(9000),:])

X20=np.concatenate([XP20[:,range(4)],XV20.reshape((9000,1)),XWV20.reshape((9000,1))],axis=1)
Ytest20=np.genfromtxt(sav_dir+'test_outsklotz.txt',delimiter=',')
XtestV20=np.genfromtxt(sav_dir+'20_involstest.txt',delimiter=',')
XtestP20=np.genfromtxt(sav_dir+'test_insklotz.txt',delimiter=',')
Xtest20=np.concatenate([XtestP20[:,range(4)],XtestV20.reshape((len(XtestV20),1)),XtestWV20.reshape((len(XtestWV20),1))],axis=1)

mx20=np.mean(X20,axis=0)
sx20=np.std(X20,axis=0)
my20=np.mean(Y20,axis=0)
sy20=np.std(Y20,axis=0)

mx30=np.mean(X30,axis=0)
sx30=np.std(X30,axis=0)
my30=np.mean(Y30,axis=0)
sy30=np.std(Y30,axis=0)

X20-=mx20
X20/=sx20
Xtest20-=mx20
Xtest20/=sx20
Y20-=my20
Y20/=sy20

X30-=mx30
X30/=sx30
Y30-=my30
Y30/=sy30

Y20=Y20[:,0].reshape((9000,1))
Y30=Y30[:,0].reshape((9000,1))

GPsav_dir20='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/klotz_training/tuning/k20WV/titsias'
GPsav_dir30='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/klotz_training/tuning/k30WV/titsias'

lens20=np.genfromtxt(GPsav_dir20+'/lengthscales'+ext+'.txt')
sigv20=np.genfromtxt(GPsav_dir20+'/signalvariance'+ext+'.txt')
likv20=np.genfromtxt(GPsav_dir20+'/likvariance'+ext+'.txt')
likv20=likv20.reshape((1))
A20=np.genfromtxt(GPsav_dir20+'/meanA'+ext+'.txt')
b20=np.genfromtxt(GPsav_dir20+'/meanb'+ext+'.txt')
Z20=np.genfromtxt(GPsav_dir20+'/inducing_ins'+ext+'.txt')
gpflowpred=np.genfromtxt(GPsav_dir20+'/predmu'+ext+'.txt')
gpflowvar=np.genfromtxt(GPsav_dir20+'/predvar'+ext+'.txt')
A20=A20.reshape((6,1))
b20=b20.reshape((1))

lens30=np.genfromtxt(GPsav_dir30+'/lengthscales'+ext+'.txt')
sigv30=np.genfromtxt(GPsav_dir30+'/signalvariance'+ext+'.txt')
likv30=np.genfromtxt(GPsav_dir30+'/likvariance'+ext+'.txt')
likv30=likv30.reshape((1))
A30=np.genfromtxt(GPsav_dir30+'/meanA'+ext+'.txt')
b30=np.genfromtxt(GPsav_dir30+'/meanb'+ext+'.txt')
Z30=np.genfromtxt(GPsav_dir30+'/inducing_ins'+ext+'.txt')

A30=A30.reshape((6,1))
b30=b30.reshape((1))

def SEkernel(x_,x,l,s): #x_ is test, x is train
    xpar1 = tf.expand_dims(x_, 1)
    xpar2 = tf.expand_dims(x, 0)
    l1_=tf.expand_dims(l,0)
    # K11=tf.multiply(s,tf.reduce_prod(tf.exp(-tf.divide(tf.square(xpar1-xpar2),2.*l1_**2)),axis=2))
    K11=tf.multiply(s,tf.exp(tf.reduce_sum(-tf.divide(tf.square(xpar1-xpar2),2.*l1_**2),axis=2)))
    return K11

def mf(x,A,b):
    return tf.tensordot(x,A, [[-1], [0]]) + b

class GPmodel():
    def __init__(self,data,kernel,pars,mean_fun,Z):
        self.X=data[0]
        self.Y=data[1]
        self.inds=Z
        ls=pars[0]
        svar=pars[1]
        self.sigma = np.sqrt(pars[2])
        self.kern=lambda x,x_: kernel(x,x_,tf.constant(ls),tf.constant(svar))
        self.mf=lambda x: mean_fun(x,pars[3],pars[4])

    def coeff_creation(self):
        nind=self.inds.shape[0]
        err=self.Y-self.mf(self.X)
        Kzz=self.kern(self.inds,self.inds)+1e-6*tf.eye(nind,nind,dtype=tf.float64)
        Kzn=self.kern(self.inds,self.X)
        L = tf.linalg.cholesky(Kzz)
        A = tf.linalg.triangular_solve(L, Kzn, lower=True) / self.sigma
        B = tf.linalg.matmul(A, A, transpose_b=True) + tf.eye(nind,nind, dtype=tf.float64)
        LB = tf.linalg.cholesky(B)
        Aerr = tf.linalg.matmul(A, err)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / self.sigma
        trm1 = tf.linalg.triangular_solve(L, tf.eye(nind,nind,dtype=tf.float64), lower=True)
        trm2 = tf.linalg.triangular_solve(LB, trm1, lower=True)
        self.coeffmu = tf.linalg.matmul(trm2, c, transpose_a=True)
        self.coeffvar1=trm1
        self.coeffvar2=trm2

    def predict(self,Xtst):
        Kxz=self.kern(Xtst,self.inds)
        Kxx=self.kern(Xtst,Xtst)
        trm1=tf.linalg.matmul(self.coeffvar1,Kxz,transpose_b=True)
        trm2=tf.matmul(self.coeffvar2,Kxz,transpose_b=True)
        mu=tf.matmul(Kxz,self.coeffmu)+self.mf(Xtst)
        var=tf.linalg.diag_part(Kxx)
        var-=tf.reduce_sum(tf.square(trm1),axis=0)
        var+=tf.reduce_sum(tf.square(trm2),axis=0)
        var+=np.square(self.sigma)
        return mu,var
pars20=[lens20,sigv20,likv20,A20,b20]
m20WV=GPmodel((X20,Y20),SEkernel,pars20,mf,Z20)

pars30=[lens30,sigv30,likv30,A30,b30]
m30WV=GPmodel((X30,Y30),SEkernel,pars30,mf,Z30)

m20WV.coeff_creation()
m30WV.coeff_creation()
