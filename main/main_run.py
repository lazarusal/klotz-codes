#main script for running the bayes klotz
# import utils
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = str(utils.pick_gpu_lowest_memory())

import sys
import tensorflow as tf
import tensorflow_probability as tfp
tfb=tfp.bijectors
import numpy as np
import matplotlib.pyplot as plt
from klotz_priors_ import *
from LVemul import *
from hmc_makers import *
import time

pr_model=sys.argv[1]
prior='unnormalized' #bayes or unnormalized
with_klotz=1 #with or without klotz prior
eps=0 #epsilon indicator
npc=5 #epsilon indicator
# corind=int(float(sys.argv[5])) #epsilon indicator
hetero=0
corind=0 # we always run with correction zero but constant bias can be used instead
dtype=tf.float64 #data type being used
pcstr=''
if prior=='unnormalized':
    if with_klotz==1:
        if eps==0:
            res_dir='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/resultsTrans/'+pcstr+'unnormalizedEps0'
        else:
            res_dir='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/resultsTrans/'+pcstr+'unnormalizedEps1'
    else:
        if eps==0:
            res_dir='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/resultsTrans/'+pcstr+'NoKlotzUnnormalizedEps0'
        else:
            res_dir='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/resultsTrans/'+pcstr+'NoKlotzUnnormalizedEps1'
else:
    if with_klotz==1:
        if prior=='bayes':
            if eps==1:
                res_dir='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/resultsTrans/'+pcstr+'bayesEps1'
            else:
                res_dir='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/resultsTrans/'+pcstr+'bayesEps0'
        else:
            if eps==1:
                res_dir='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/resultsTrans/'+pcstr+'multiEps1'
            else:
                res_dir='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/resultsTrans/'+pcstr+'multiEps0'
    else:
        if eps==1:
            res_dir='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/resultsTrans/'+pcstr+'NoKlotzNormalizedEps1'
        else:
            res_dir='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/resultsTrans/'+pcstr+'NoKlotzNormalizedEps0'

if corind==1:
    res_dir=res_dir+'/corrected'
else:
    res_dir=res_dir+'/uncorrected'

chaindir=res_dir+'/chains'
plotdir=res_dir+'/plots'

if not os.path.exists(chaindir):
    os.makedirs(chaindir)
if not os.path.exists(plotdir):
    os.makedirs(plotdir)


if prior=='unnormalized':
    sav_d='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/test_data/unnormalized_real' #sav_dir
else:
    sav_d='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/test_data/normalized_real' #sav_dir

tst_x,tst_y=load_data(sav_d,'Eps'+str(eps))

tst_y-=my
tst_y/=sy
if pr_model=='Gauss':
    res_dir='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/results/DeltaVsGauss/Gauss'
    def prior_log_prob_fn(pars,sig,V0,V8,WV,V8_unc=0):
        return unnormalizedeps0(pars,V0,WV,V8,V8_unc)+tf.reduce_sum(tfp.distributions.InverseGamma(tf.cast(0.001,tf.float64),tf.cast(0.001,tf.float64)).log_prob(sig))
elif pr_model=='EmpiNew':
    res_dir='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/results/DeltaVsGauss/UpdatedEmpi'
    def prior_log_prob_fn(pars,sig,V0,V8,WV,V8_unc=0):
        return unnormalizedeps0NEWNEW(pars,V0,WV,V8,V8_unc)+tf.reduce_sum(tfp.distributions.InverseGamma(tf.cast(0.001,tf.float64),tf.cast(0.001,tf.float64)).log_prob(sig))
else:
    res_dir='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/results/DeltaVsGauss/FirstOrderDelta'
    def prior_log_prob_fn(pars,sig,V0,V8,WV,V8_unc=0):
        return unnormalizedeps0NEW(pars,V0,WV,V8,V8_unc)+tf.reduce_sum(tfp.distributions.InverseGamma(tf.cast(0.001,tf.float64),tf.cast(0.001,tf.float64)).log_prob(sig))
# res_dir='/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/Results2021/'+pcstr+'unnormalizedEps1'

chaindir=res_dir+'/chains'
plotdir=res_dir+'/plots'

if not os.path.exists(chaindir):
    os.makedirs(chaindir)
if not os.path.exists(plotdir):
    os.makedirs(plotdir)

vols=np.genfromtxt('/xlwork4/2026068l/PhD/projects/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/test_data/test_vols.txt',delimiter=',').reshape((11,2))

# prior_log_prob_fn=select_prior(with_klotz,prior,eps)

def log_prob_fn(pars,sig,Y,inds,corr,geom): #Likelihood
    pred=tf.reshape(model(pars,geom),[25])
    pred+=corr
    return tf.reduce_sum(tfd.Normal(tf.gather(pred,inds),tf.sqrt(sig)).log_prob(tf.gather(Y,inds)))

def target_log_prob_fn(thets,sig,Y,inds,corr,V0,V8,WV,geom): #loglik+logprior
    pr_prob=prior_log_prob_fn(thets,sig,V0,V8,WV)
    lik=log_prob_fn(thets,sig,Y,inds,corr,geom)
    return lik+pr_prob

sigmoid =tfb.Shift(tf.cast(0.1,tf.float64))(tfb.Scale(tf.cast(4.9,tf.float64))(tfb.Reciprocal()(
    tfb.Shift(tf.cast(1.,tf.float64))(
      tfb.Exp()(
        tfb.Scale(tf.cast(-1.,tf.float64)))))))

bij=[sigmoid,tfb.Exp()] #sample in unconstrained space

nburn=10000
nsamp=10000

#need new getter functions to access inner_kernel


# def step_size_setter_fn(kernel_results, new_step_size):
#     pars=kernel_results.inner_results
#     pars=pars._replace(
#             step_size=new_step_size)
#     return kernel_results._replace(
#             inner_results=pars)
# #
#
# def step_size_getter_fn(kernel_results):
#     return tf.cast(kernel_results.inner_results.step_size, dtype)
#
#
# def log_accept_prob_getter_fn(kernel_results):
#     return kernel_results.inner_results.log_accept_ratio


def nuts_adaptive_kernel(y,inds,corr,V0,V8,WV,geom):
    nuts=tfp.mcmc.NoUTurnSampler(
    target_log_prob_fn=lambda x,sig: target_log_prob_fn(x,sig,y,inds,corr,V0,V8,WV,geom),
    step_size=tf.cast(0.01,tf.float64),max_tree_depth=10)

    TransNuts= tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=nuts,
    bijector=bij)

    return tfp.mcmc.DualAveragingStepSizeAdaptation(
    TransNuts,
    num_adaptation_steps=int(nburn*0.8),
    step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
            inner_results=pkr.inner_results._replace(step_size=new_step_size)),
    step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
    log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
    target_accept_prob=tf.cast(.75, dtype),
    decay_rate=tf.cast(.75, dtype)
)


@tf.function(experimental_compile=True)
def run_nuts(y,inds,corr,V0,V8,WV,geom,init_state):
    kern=nuts_adaptive_kernel(y,inds,corr,V0,V8,WV,geom)
    return tfp.mcmc.sample_chain(
    num_results=nsamp,
    num_burnin_steps=nburn,
    current_state=init_state,
    kernel=kern
    )

for g_ in range(6):
    V0=tf.constant([[vols[g_,0]]])
    WV=tf.constant([[vols[g_,1]]])
    for p in range(5):
        Ytst=tf.constant(tst_y[g_*5+p,:])
        V8=tf.constant(Ytst[0]*sy[0]+my[0])
        samps=[]
        sigs=[]
        betas=[]
        samps_burn=[]
        sigs_burn=[]
        states=[]
        start_time = time.time()
        sig=np.float64(1.) #initial noise standard devation
        Xtst=tst_x[g_*5+p,:]

        state=[tf.constant(Xtst.reshape((1,4))),tf.constant(sig.reshape((1,1)))]

        if npc==10:
            geom=proj[g_,:].reshape((1,10))
        elif npc==5:
            geom=proj[g_,:].reshape((1,5))
        else:
            geom=np.zeros((1,5),dtype=np.float64)


        corr=tf.constant(np.float64(0.))

        inds=list(range(25))

        states,extra=run_nuts(Ytst,inds,corr,V0,V8,WV,geom,state)
        chain=states[0].numpy()
        chain=chain.reshape((nsamp,4))

        # sigs=states[1].numpy()
        # sigs=sigs.reshape((nsamp,1))
        # fig, axs = plt.subplots(2, 2)
        # axs[0,0].plot(range(chain.shape[0]),chain[:,0],alpha=0.5,c='steelblue')
        # axs[0,0].set_ylim([0.1,5])
        # axs[0,1].plot(range(chain.shape[0]),chain[:,1],alpha=0.5,c='red')
        # axs[0,1].set_ylim([0.1,5])
        # axs[1,0].plot(range(chain.shape[0]),chain[:,2],alpha=0.5,c='forestgreen')
        # axs[1,0].set_ylim([0.1,5])
        # axs[1,1].plot(range(chain.shape[0]),chain[:,3],alpha=0.5,c='orange')
        # axs[1,1].set_ylim([0.1,5])
        # axs[0,0].axhline(1,c='steelblue',linestyle='--',linewidth=2)
        # axs[0,1].axhline(Xtst[1],c='red',linestyle='--',linewidth=2)
        # axs[1,0].axhline(Xtst[2],c='forestgreen',linestyle='--',linewidth=2)
        # axs[1,1].axhline(1.,c='orange',linestyle='--',linewidth=2)
        # axs[0,0].set_title(r'$\theta_1$')
        # axs[0,1].set_title(r'$\theta_2$')
        # axs[1,0].set_title(r'$\theta_3$')
        # axs[1,1].set_title(r'$\theta_4$')
        # fig.tight_layout()
        # plt.savefig(plotdir+'/GNUTSthetasgeo'+str(g_+1)+'par'+str(p)+'.png')
        # plt.close()
        # plt.plot(range(chain.shape[0]),sigs,alpha=0.5,c='black')
        # plt.ylabel(r'$\sigma^2$',fontsize=14)
        # plt.xlabel('Iteration',fontsize=14)
        # fig.tight_layout()
        # plt.savefig(plotdir+'/GNUTSsiggeo'+str(g_+1)+'par'+str(p)+'.png')
        # plt.close()
        np.savetxt(chaindir+'/GNUTSthetasgeo'+str(g_+1)+'par'+str(p)+'.txt',chain)
