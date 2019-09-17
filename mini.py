from g_factors3 import gen_f
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import burr
import pandas as pd
import numpy as np
def gen_v():
    data = pd.read_csv("data.csv")
    data[data==' ']=np.nan
    data=data.dropna()
    data.head(5)
    data_E=data.iloc[0:56,:]
    data_T=data.iloc[56::,:]
    stress=np.int8(data_E.iloc[:,31])
    tired=np.int8(data_E.iloc[:,32])
    st=np.int8(data_E.iloc[:,31:33])
    choice3=np.int8(data_E.iloc[:,37])
    choice4=np.int8(data_E.iloc[:,38])
    tend=np.int8(choice3)+np.int8(choice4)
    risk=np.int8(data_E.iloc[:,34])
    delay=np.int8(data_E.iloc[:,39])
    fst=np.zeros((3,3))
    for i in range (0,3):
        for j in range (0,3):
            fst[i,j]=np.array([i for i in np.where(st[:,0]==i)[0] if i in np.where(st[:,1]==j)[0]]).size
    fst/=fst.sum()
    frd=np.zeros((2,2))
    for i in range (0,2):
        for j in range (0,2):
            frd[i,j]=np.array([i for i in np.where(risk==i+1)[0] if i in np.where(delay==j+1)[0]]).size
    frd/=frd.sum()
    frd
    time=np.float32(data_E.iloc[:,1:11])
    mu=np.zeros(10)
    std=np.zeros(10)
    for i in range (0,10):
        mu[i],std[i]=norm.fit(time[:,i])
    farmiliar=np.float32(data_E.iloc[:,21:31])
    mu_f=np.zeros(10)
    std_f=np.zeros(10)
    for i in range (0,10):
        mu_f[i],std_f[i]=norm.fit(farmiliar[:,i])
    m1=np.load('m1.npy')
    m2=np.load('m2.npy')
    m3=np.load('m3.npy')
    m4=np.load('m4.npy')
    f=data_E.iloc[:,21:31]
    f_e=np.zeros((data_E.shape[0],10))
    rv1=burr(m1[0], m1[1], m1[2], m1[3])
    rv2=burr(m2[0], m2[1], m2[2], m2[3])
    rv3=burr(m3[0], m3[1], m3[2], m3[3])
    rv4=burr(m4[0], m4[1], m4[2], m4[3])
    x=np.arange(0,4000)
    f_e[np.where(f=='0')] = 0
    f_e[np.where(f=='1')] = x.dot(rv1.pdf(x))
    f_e[np.where(f=='2')] = x.dot(rv2.pdf(x))
    f_e[np.where(f=='3')] = x.dot(rv3.pdf(x))
    f_e[np.where(f=='4')] = x.dot(rv4.pdf(x))
    mu_fe=np.zeros(10)
    std_fe=np.zeros(10)
    for i in range (0,10):
        mu_fe[i],std_fe[i]=norm.fit(f_e[:,i])
    v=[fst,frd,mu,std,mu_fe,std_fe,mu_f,std_f]
    return v,tend,stress,tired,risk,delay,time,farmiliar,f_e

def fun(v,i,stress,tired,risk,delay,time,farmiliar,f_e):
    fst,frd,mu,std,mu_fe,std_fe,mu_f,std_f=v
    #print(frd)
    tend_p=fst[stress[i],tired[i]]*frd[risk[i]-1,delay[i]-1]
    for i1 in range (0,10):
        tend_p*=norm.pdf(time[i,i1],mu[i1],std[i1])
    for i1 in range (0,10):
        tend_p*=norm.pdf(farmiliar[i,i1],mu_f[i1],std_f[i1])
    for i1 in range (0,10):
        tend_p*=norm.pdf(f_e[i,i1],mu_fe[i1],std_fe[i1])
    return tend_p

def err(v,tend,stress,tired,risk,delay,time,farmiliar,f_e):
    error=0
    p=np.zeros(56)
    for i in range(0,56):
        p[i]=fun(v,i,stress,tired,risk,delay,time,farmiliar,f_e)
    p=p/p.sum()
    #mu_tend,std_tend=norm.fit(tend)
    for i in range(0,56):
        if tend[i]>20:
            error+=np.abs(1-p[i])
        else:
            error+=np.abs(0-p[i])
    return error

def update(m,load):
    v,tend,stress,tired,risk,delay,time,farmiliar,f_e=gen_v()
    if load==1:
        v=np.load('v.npy')
    fst,frd,mu,std,mu_fe,std_fe,mu_f,std_f=v
    function1 = lambda fst: err(v,tend,stress,tired,risk,delay,time,farmiliar,f_e)
    function2 = lambda frd: err(v,tend,stress,tired,risk,delay,time,farmiliar,f_e)
    function3 = lambda mu: err(v,tend,stress,tired,risk,delay,time,farmiliar,f_e)
    function4 = lambda std: err(v,tend,stress,tired,risk,delay,time,farmiliar,f_e)
    function5 = lambda mu_fe: err(v,tend,stress,tired,risk,delay,time,farmiliar,f_e)
    function6 = lambda std_fe: err(v,tend,stress,tired,risk,delay,time,farmiliar,f_e)
    function7 = lambda mu_f: err(v,tend,stress,tired,risk,delay,time,farmiliar,f_e)
    function8 = lambda std_f: err(v,tend,stress,tired,risk,delay,time,farmiliar,f_e)
    f=[function1,function2,function3,function4,function5,function6,function7,function8]
    for i in range (0,m):
        k=np.random.randint(0,7)
        t=32
        if i%100==0:
            t/=2
        v[k]=minimize(f[k],v[k],method='Nelder-Mead', tol=t)['x'].reshape(v[k].shape)
        if i%10==0:
            np.save('v.npy',v)