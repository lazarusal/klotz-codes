import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd=tfp.distributions
dataEX=np.genfromtxt("C:/Users/alanl/OneDrive - University of Glasgow/PhD/projects/NEW_ORGANISATION/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/data/plot_data.txt",delimiter=',')

klotz_pars=pickle.load(open("C:/Users/alanl/OneDrive - University of Glasgow/PhD/projects/NEW_ORGANISATION/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/DeltaVsMC/KlotzPars.p",'rb'))    
klotz_pars=[x[::10] for z in klotz_pars]

V0=78.
V8=110

klotz_parsunnorm=np.genfromtxt('C:/Users/alanl/OneDrive - University of Glasgow/PhD/projects/NEW_ORGANISATION/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/data/unnorm_kpars.txt',delimiter=',')
klotz_stdsunnorm=np.genfromtxt('C:/Users/alanl/OneDrive - University of Glasgow/PhD/projects/NEW_ORGANISATION/parameter-geometry-emulator_pca/5pc_emulator/klotz/pythoncode/data/unnorm_kstds.txt',delimiter=',')


alpm=np.mean(klotz_pars[0])
betm=np.mean(klotz_pars[1])
alps=np.std(klotz_pars[0])
bets=np.std(klotz_pars[1])


samps=samps[5000::5,:]
MCsamps=samps[:,1]*(V8-V0)/np.log(8./samps[:,0])+V0

dvda=betm*(V8-V0)/(alpm*np.log(8/alpm)**2)
dvdb=(V8-V0)/np.log(8/alpm)
dvdba=(V8-V0)/(alpm*np.log(8/alpm)**2)

V30K=betm*(V8-V0)/np.log(8./alpm)+V0
sig2K=dvda**2*alps**2+dvdb**2*bets**2-2*dvdb*dvda*0.002

DELTAsamps=np.random.normal(V30K,np.sqrt(sig2K),[100000,1])

fig, axs = plt.subplots(1,1, figsize=(15, 8), facecolor='w', edgecolor='k')
sns.kdeplot(DELTAsamps[:,0],ax=axs)
sns.kdeplot(MCsamps,ax=axs)
axs.set_xlim([110,150])
