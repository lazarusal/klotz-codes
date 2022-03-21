#contains functions for the Klotz priors Bayes, multitask and unnormalized
#calls some gpflow functions for V20 and V30 emulators

import tensorflow as tf
import tensorflow_probability as tfp
tfd=tfp.distributions
import gpflow
import pickle
import numpy as np
from klotz_gps import *

sav_dir='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/10pc_emulator/cartesian_10pc_emulation_data/'

#now we define the Bayes prior for Klotz, first reading in MCMC samples of parameters


klotz_pars_multi=np.genfromtxt('/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/data/klotz_pars_multi.txt',delimiter=',')
klotz_data=np.genfromtxt('/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/data/plot_data.txt',delimiter=',')
klotz_parssd_multi=np.std(klotz_pars_multi[range(30000,klotz_pars_multi.shape[0]),:],axis=0)
klotz_pars_multi=np.mean(klotz_pars_multi[range(30000,klotz_pars_multi.shape[0]),:],axis=0)


klotz_pars=np.genfromtxt('/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/data/klotz_pars.txt',delimiter=',')
klotz_data=np.genfromtxt('/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/data/plot_data.txt',delimiter=',')
klotz_parssd=np.std(klotz_pars[range(30000,klotz_pars.shape[0]),:],axis=0)
klotz_pars=np.mean(klotz_pars[range(30000,klotz_pars.shape[0]),:],axis=0)

klotz_parsunnorm=np.genfromtxt('/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/data/unnorm_kpars.txt',delimiter=',')
klotz_stdsunnorm=np.genfromtxt('/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/data/unnorm_kstds.txt',delimiter=',')


sig2=np.exp(klotz_parsunnorm[2]+klotz_parsunnorm[3]*8.+klotz_parsunnorm[4]*8.**2)
alpm=klotz_parsunnorm[0]
betm=klotz_parsunnorm[1]
alps=klotz_stdsunnorm[0]
bets=klotz_stdsunnorm[1]

def bayes_prior(x,V0,WV): #defining the prior for the Bayes method
    X20=tf.concat((x,V0,WV),axis=1)
    X30=tf.concat((x,V0,WV),axis=1)
    X20-=mx20
    X20/=sx20
    X30-=mx30
    X30/=sx30
    mu20,var20=m20WV.predict(tf.reshape(X20,[1,6]))
    mu30,var30=m30WV.predict(tf.reshape(X30,[1,6]))
    mu20=mu20*sy20[0]+my20[0]
    mu30=mu30*sy30[0]+my30[0]
    var20=var20*sy20[0]**2
    var30=var30*sy30[0]**2
    sig2K=np.exp(klotz_pars[1]+klotz_pars[2]*2/3+klotz_pars[3]*4/9)
    sig2E=tf.square(tf.divide(1.,mu30-V0[0][0]))*var20+tf.square(tf.divide(mu20-V0[0][0],tf.square(mu30-V0[0][0])))*var30
    return tf.reduce_sum(tfd.Normal(1/klotz_pars[0]*np.log(2./3.)+1.,tf.sqrt(sig2K+sig2E)).log_prob(tf.divide(mu20-V0[0][0],mu30-V0[0][0])))

def bayes_priorEps0(x,V0,WV): #defining the prior for the Bayes method
    X20=tf.concat((x,V0,WV),axis=1)
    X30=tf.concat((x,V0,WV),axis=1)
    X20-=mx20
    X20/=sx20
    X30-=mx30
    X30/=sx30
    mu20,var20=m20WV.predict(tf.reshape(X20,[1,6]))
    mu30,var30=m30WV.predict(tf.reshape(X30,[1,6]))
    mu20=mu20*sy20[0]+my20[0]
    mu30=mu30*sy30[0]+my30[0]
    var20=var20*sy20[0]**2
    var30=var30*sy30[0]**2
    sig2K=(np.log(2./3.)/klotz_pars[0]**2)**2*klotz_parssd[0]**2
    sig2E=tf.square(tf.divide(1.,mu30-V0[0][0]))*var20+tf.square(tf.divide(mu20-V0[0][0],tf.square(mu30-V0[0][0])))*var30
    return tf.reduce_sum(tfd.Normal(1/klotz_pars[0]*np.log(2./3.)+1.,tf.sqrt(sig2K+sig2E)).log_prob(tf.divide(mu20-V0[0][0],mu30-V0[0][0])))


def bayes_prior23Eps0(x,V0,WV): #defining the prior for the Bayes method
    X20=tf.concat(([[1.]],x,[[1.]],V0,WV),axis=1)
    X30=tf.concat(([[1.]],x,[[1.]],V0,WV),axis=1)
    X20-=mx20
    X20/=sx20
    X30-=mx30
    X30/=sx30
    mu20,var20=m20WV.predict(tf.reshape(X20,[1,6]))
    mu30,var30=m30WV.predict(tf.reshape(X30,[1,6]))
    mu20=mu20*sy20[0]+my20[0]
    mu30=mu30*sy30[0]+my30[0]
    var20=var20*sy20[0]**2
    var30=var30*sy30[0]**2
    sig2K=(np.log(2./3.)/klotz_pars[0]**2)**2*klotz_parssd[0]**2
    sig2E=tf.square(tf.divide(1.,mu30-V0[0][0]))*var20+tf.square(tf.divide(mu20-V0[0][0],tf.square(mu30-V0[0][0])))*var30
    return tf.reduce_sum(tfd.Normal(1/klotz_pars[0]*np.log(2./3.)+1.,tf.sqrt(sig2K+sig2E)).log_prob(tf.divide(mu20-V0[0][0],mu30-V0[0][0])))


def preds(x,V0,WV): #defining the prior for the Bayes method
    X20=tf.concat((x,V0,WV),axis=1)
    X30=tf.concat((x,V0,WV),axis=1)

    X20-=mx20
    X20/=sx20
    X30-=mx30
    X30/=sx30
    mu20,var20=m20WV.predict(tf.reshape(X20,[1,6]))
    mu30,var30=m30WV.predict(tf.reshape(X30,[1,6]))
    mu20=mu20*sy20[0]+my20[0]
    mu30=mu30*sy30[0]+my30[0]
    # var20=var20*sy20[0]**2
    # var30=var30*sy30[0]**2
    # sig2K=(np.log(2./3.)/klotz_pars[0]**2)**2*klotz_parssd[0]**2
    # sig2E=tf.square(tf.divide(1.,mu30-V0[0][0]))*var20+tf.square(tf.divide(mu20-V0[0][0],tf.square(mu30-V0[0][0])))*var30
    # return tf.reduce_sum(tfd.Normal(1/klotz_pars[0]*np.log(2./3.)+1.,tf.sqrt(sig2K+sig2E)).log_prob(tf.divide(mu20-V0[0][0],mu30-V0[0][0])))
    return [mu20,mu30]


def multi_taskEps1(x,V0,WV): #defining the ex-vivo likeliihood for multi task approach
    #sigex=np.exp(x[4]+x[5]*klotz_data[:,0]+x[6]*np.square(klotz_data[:,0]))
    sigex=np.exp(klotz_pars_multi[2]+klotz_pars_multi[3]*(klotz_data[:,0])+klotz_pars_multi[4]*(klotz_data[:,0])**2)
    X20=tf.concat((x[:4],V0,WV),axis=1)
    X30=tf.concat((x[:4],V0,WV),axis=1)
    X20-=mx20
    X20/=sx20
    X30-=mx30
    X30/=sx30
    mu20,var20=m20WV.predict(tf.reshape(X20,[1,6]))
    mu30,var30=m30WV.predict(tf.reshape(X30,[1,6]))
    mu20=mu20*sy20[0]+my20[0]
    mu30=mu30*sy30[0]+my30[0]
    var20=var20*sy20[0]**2
    var30=var30*sy30[0]**2
    alp=tf.exp(np.log(3./2.)*tf.divide(mu30-V0,mu20-mu30))
    bet=np.log(2./3.)*tf.divide(mu30-V0,mu20-mu30)
    fex=tf.multiply(alp,tf.exp(bet*klotz_data[:,0]))
    # likex=tfd.Normal(fex,sigex).log_prob(klotz_data[:,1])
    dalp20=alp*np.log(2./3.)*tf.divide(mu30-V0,tf.square(mu20-mu30))
    dbet20=tf.divide(bet,mu30-mu20)
    dalp30=alp*np.log(3./2.)*tf.divide(mu20-V0,tf.square(mu20-mu30))
    dbet30=np.log(2./3.)*tf.divide(mu20-V0,tf.square(mu20-mu30))
    df20=fex/alp*dalp20+tf.multiply(klotz_data[:,0],fex)*dbet20
    df30=fex/alp*dalp30+tf.multiply(klotz_data[:,0],fex)*dbet30

    sigf2=tf.square(df20)*var20+tf.square(df30)*var30

    return tf.reduce_sum(tfd.Normal(fex,tf.sqrt(sigf2+sigex)).log_prob(klotz_data[:,1]/30))

def multi_taskEps0(x,V0,WV): #defining the ex-vivo likeliihood for multi task approach
    # sigex2=tf.exp(tf.gather(x,4,axis=1)+tf.gather(x,5,axis=1)*klotz_data[:,0]+tf.gather(x,6,axis=1)*klotz_data[:,0]**2)
    sigex2=tf.exp(tf.gather(x,4,axis=1))
    X20=tf.concat((tf.gather(x,[0,1,2,3],axis=1),V0,WV),axis=1)
    X30=tf.concat((tf.gather(x,[0,1,2,3],axis=1),V0,WV),axis=1)
    X20-=mx20
    X20/=sx20
    X30-=mx30
    X30/=sx30
    mu20,var20=m20WV.predict(tf.reshape(X20,[1,6]))
    mu30,var30=m30WV.predict(tf.reshape(X30,[1,6]))
    mu20=mu20*sy20[0]+my20[0]
    mu30=mu30*sy30[0]+my30[0]
    var20=var20*sy20[0]**2
    var30=var30*sy30[0]**2
    alp=tf.exp(np.log(3./2.)*tf.divide(mu30-V0,mu20-mu30))
    bet=np.log(2./3.)*tf.divide(mu30-V0,mu20-mu30)
    fex=tf.multiply(alp,tf.exp(bet*klotz_data[:,0]))
    # likex=tfd.Normal(fex,sigex).log_prob(klotz_data[:,1])
    dalp20=alp*np.log(2./3.)*tf.divide(mu30-V0,tf.square(mu20-mu30))
    dbet20=tf.divide(bet,mu30-mu20)
    dalp30=alp*np.log(3./2.)*tf.divide(mu20-V0,tf.square(mu20-mu30))
    dbet30=np.log(2./3.)*tf.divide(mu20-V0,tf.square(mu20-mu30))
    df20=fex/alp*dalp20+tf.multiply(klotz_data[:,0],fex)*dbet20
    df30=fex/alp*dalp30+tf.multiply(klotz_data[:,0],fex)*dbet30
    sigf2=tf.square(df20)*var20+tf.square(df30)*var30
    return tf.reduce_sum(tfd.Normal(klotz_data[:,1]/30.,tf.sqrt(sigf2+sigex2)).log_prob(fex))

def multi_taskPLOT(x,V0,WV): #defining the ex-vivo likeliihood for multi task approach
    X20=tf.concat((tf.gather(x,[0,1,2,3],axis=1),np.tile(V0,[x.shape[0],1]),np.tile(WV,[x.shape[0],1])),axis=1)
    X30=tf.concat((tf.gather(x,[0,1,2,3],axis=1),np.tile(V0,[x.shape[0],1]),np.tile(WV,[x.shape[0],1])),axis=1)
    X20-=mx20
    X20/=sx20
    X30-=mx30
    X30/=sx30
    mu20,var20=m20WV.predict(tf.reshape(X20,[1,6]))
    mu30,var30=m30WV.predict(tf.reshape(X30,[1,6]))
    mu20=mu20*sy20[0]+my20[0]
    mu30=mu30*sy30[0]+my30[0]
    var20=var20*sy20[0]**2
    var30=var30*sy30[0]**2
    alp=tf.exp(np.log(3./2.)*tf.divide(mu30-V0,mu20-mu30))
    bet=np.log(2./3.)*tf.divide(mu30-V0,mu20-mu30)
    return alp,bet
    # return tf.reduce_sum(tfd.Normal(fex,10000.).log_prob(klotz_data[:,1]))

def unnormalizedeps1(x,V0,WV,V8,V8_unc):
    X30=tf.concat((x ,V0,WV),axis=1)
    V0=V0[0][0]
    X30-=mx30
    X30/=sx30
    mu30,var30=m30WV.predict(tf.reshape(X30,[1,6]))
    mu30=mu30*sy30[0]+my30[0]
    var30=var30*sy30[0]**2
    dvda=betm*(V8-V0)/(alpm*np.log(8/alpm)**2)
    dvdb=(V8-V0)/np.log(8/alpm)
    dvde=-betm**2*(V8-V0)/(np.log(8/alpm)**2)
    V30K=betm*(V8-V0)/np.log(8./alpm)+V0
    sig2K=dvda**2*alps**2+dvdb**2*bets**2+dvde**2*sig2
    if V8_unc!=0:
        dvdm=betm/np.log(8./alpm)
        sig2K+=np.square(dvdm)*V8_unc
    return tf.reduce_sum(tfd.Normal(V30K,tf.sqrt(sig2K+var30)).log_prob(mu30))


def unnormalizedeps0(x,V0,WV,V8,V8_unc):
    X30=tf.concat((x ,V0,WV),axis=1)
    V0=V0[0][0]
    # V0=V0[0]
    X30-=mx30
    X30/=sx30
    # mu30,var30=m30WV.predict(tf.reshape(X30,[1,6]))
    mu30,var30=m30WV.predict(X30)
    mu30=mu30*sy30[0]+my30[0]
    var30=var30*sy30[0]**2
    V30K=betm*(V8-V0)/np.log(8./alpm)+V0
    dvda=betm*(V8-V0)/(alpm*np.log(8/alpm)**2)
    dvdb=(V8-V0)/np.log(8/alpm)
    sig2K=dvda**2*alps**2+dvdb**2*bets**2
    if V8_unc!=0:
        dvdm=betm/np.log(8./alpm)
        sig2K+=np.square(dvdm)*V8_unc
    return tf.reduce_sum(tfd.Normal(V30K,tf.sqrt(sig2K+var30)).log_prob(mu30))


def unnormalizedeps0NEWNEW(x,V0,WV,V8,V8_unc):
    X30=tf.concat((x ,V0,WV),axis=1)
    V0=V0[0][0]
    # V0=V0[0]
    X30-=mx30
    X30/=sx30
    # mu30,var30=m30WV.predict(tf.reshape(X30,[1,6]))
    mu30,var30=m30WV.predict(X30)
    mu30=mu30*sy30[0]+my30[0]
    var30=var30*sy30[0]**2
    # V30K=betm*(V8-V0)/np.log(8./alpm)+V0
    V30K=(V8-V0)/(np.log(8./30.)/klotz_pars[0]+1)+V0
    # dvda=betm*(V8-V0)/(alpm*np.log(8/alpm)**2)
    dvdb=(V8-V0)*np.log(8./30.)/(np.log(8/30)+klotz_pars[0])**2
    sig2K=dvdb**2*klotz_parssd[0]**2
    if V8_unc!=0:
        dvdm=betm/np.log(8./alpm)
        sig2K+=np.square(dvdm)*V8_unc
    return tf.reduce_sum(tfd.Normal(V30K,tf.sqrt(sig2K+var30)).log_prob(mu30))


def unnormalizedeps0NEW(x,V0,WV,V8,V8_unc):
    X30=tf.concat((x ,V0,WV),axis=1)
    V0=V0[0][0]
    # V0=V0[0]
    X30-=mx30
    X30/=sx30
    mu30,var30=m30WV.predict(X30)
    mu30=mu30*sy30[0]+my30[0]
    var30=var30*sy30[0]**2
    V30K=betm*(V8-V0)/np.log(8./alpm)+V0
    dvda=betm*(V8-V0)/(alpm*np.log(8/alpm)**2)
    dvdb=(V8-V0)/np.log(8/alpm)
    dvdba=(V8-V0)/(alpm*np.log(8/alpm)**2)
    sig2K=dvda**2*alps**2+dvdb**2*bets**2-2*dvdb*dvda*0.002
    if V8_unc!=0:
        dvdm=betm/np.log(8./alpm)
        sig2K+=np.square(dvdm)*V8_unc
    return tf.reduce_sum(tfd.Normal(V30K,tf.sqrt(sig2K+var30)).log_prob(mu30))

def unnormalizedeps1NEW(x,V0,WV,V8,V8_unc):
    X30=tf.concat((x ,V0,WV),axis=1)
    V0=V0[0][0]
    # V0=V0[0]
    X30-=mx30
    X30/=sx30
    # mu30,var30=m30WV.predict(tf.reshape(X30,[1,6]))
    mu30,var30=m30WV.predict(X30)
    mu30=mu30*sy30[0]+my30[0]
    var30=var30*sy30[0]**2
    V30K=betm*(V8-V0)/np.log(8./alpm)+V0
    dvda=betm*(V8-V0)/(alpm*np.log(8/alpm)**2)
    dvdb=(V8-V0)/np.log(8/alpm)
    dvde=-betm**2*(V8-V0)/(np.log(8/alpm)**2)
    #dvdba=(V8-V0)/(alpm*np.log(8/alpm)**2)
    sig2K=dvda**2*alps**2+dvdb**2*bets**2-2*dvdb*dvda*0.002+dvde**2*sig2
    if V8_unc!=0:
        dvdm=betm/np.log(8./alpm)
        sig2K+=np.square(dvdm)*V8_unc
    return tf.reduce_sum(tfd.Normal(V30K,tf.sqrt(sig2K+var30)).log_prob(mu30))

def select_prior(with_klotz,prior,eps):
    if with_klotz==0:
        def prior_log_prob_fn(pars,sig,V0,V8,WV,V8_unc=0):
            return tf.reduce_sum(tfp.distributions.InverseGamma(tf.cast(0.001,tf.float64),tf.cast(0.001,tf.float64)).log_prob(sig))  #tf.reduce_sum(tfp.distributions.Uniform(tf.cast(0.1,tf.float64),tf.cast(5.,tf.float64)).log_prob(pars))

    elif prior=='unnormalized' and eps==1:
        def prior_log_prob_fn(pars,sig,V0,V8,WV,V8_unc=0):
            return unnormalizedeps1(pars,V0,WV,V8,V8_unc)+tf.reduce_sum(tfp.distributions.InverseGamma(tf.cast(0.001,tf.float64),tf.cast(0.001,tf.float64)).log_prob(sig))

    elif prior=='unnormalized' and eps==0:
        def prior_log_prob_fn(pars,sig,V0,V8,WV,V8_unc=0):
            return unnormalizedeps0(pars,V0,WV,V8,V8_unc)+tf.reduce_sum(tfp.distributions.InverseGamma(tf.cast(0.001,tf.float64),tf.cast(0.001,tf.float64)).log_prob(sig))

    elif prior=='bayes' and eps==1:
        def prior_log_prob_fn(pars,V0,V8,WV,V8_unc=0):
            return bayes_prior(tf.gather(pars,[0,1,2,3],axis=1),V0,WV)+tf.reduce_sum(tfp.distributions.InverseGamma(tf.cast(0.001,tf.float64),tf.cast(0.001,tf.float64)).log_prob(tf.gather(pars,4,axis=1)))
    elif prior=='bayes' and eps==0:
        def prior_log_prob_fn(pars,V0,V8,WV,V8_unc=0):
            return bayes_priorEps0(tf.gather(pars,[0,1,2,3],axis=1),V0,WV)+tf.reduce_sum(tfp.distributions.InverseGamma(tf.cast(0.001,tf.float64),tf.cast(0.001,tf.float64)).log_prob(tf.gather(pars,4,axis=1)))
    # elif prior=='multi' and eps==0:
    #     def prior_log_prob_fn(pars,V0,V8,WV,V8_unc=0):
    #         return multi_taskEps0(pars,V0,WV)+tf.reduce_sum(tfp.distributions.Uniform(tf.cast(0.1,tf.float64),tf.cast(5.,tf.float64)).log_prob(tf.gather(pars,[0,1,2,3],axis=1)))+tf.reduce_sum(tfp.distributions.Uniform(tf.cast(-100.,tf.float64),tf.cast(100.,tf.float64)).log_prob(tf.gather(pars,[4],axis=1)))
    # else:
    #     def prior_log_prob_fn(pars,V0,V8,WV,V8_unc=0):
    #         return multi_taskEps1(pars,V0,WV)+tf.reduce_sum(tfp.distributions.Uniform(tf.cast(0.1,tf.float64),tf.cast(5.,tf.float64)).log_prob(pars))

    return prior_log_prob_fn

#####prior for synthetic test generation
