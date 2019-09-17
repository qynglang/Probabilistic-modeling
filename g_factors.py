from scipy.stats import burr
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from distribut import best_fit_distribution
import numpy as np
def gen_f(data_E,update=False):
        
# polynomial weight
    w0=np.load('w0.npy')
# f_e weight (familarity_exploration)
    w=np.load('w.npy')
# f weight (familarity_exploration)
    w1=np.load('w1.npy')
# risk and delay
    k=np.load('k.npy')
    w2=np.load('w2.npy')
#time
    w_t=np.load('w_t.npy')
# burr
    m1=np.load('m1.npy')
    m2=np.load('m2.npy')
    m3=np.load('m3.npy')
    m4=np.load('m4.npy')
    if update==1:
        np.save('w0_old.npy',w0)
        np.save('w_old.npy',w)
        np.save('w1_old.npy',w1)
        np.save('w2_old.npy',w2)
        np.save('k_old.npy',k)
        np.save('w_t_old.npy',w_t)
        m1=np.save('m1.npy',m1)
        m2=np.save('m2.npy',m2)
        m3=np.save('m3.npy',m3)
        m4=np.save('m4.npy',m4)
        model_update(data_E)
    stress=np.int8(data_E.iloc[:,31])
    tired=np.int8(data_E.iloc[:,32])
    st=np.int8(data_E.iloc[:,31:33])
    choice3=data_E.iloc[:,37]
    choice4=data_E.iloc[:,38]
    tend=np.int8(choice3)+np.int8(choice4)
    x1=stress
    x2=tired
    f1=w0[0]+w0[1]*np.int8(x1)+w0[2]*np.int8(x2)+w0[3]*np.int8(x1)**2+w0[4]*np.int8(x1)*np.int8(x2)+w0[5]*np.int8(x2)**2
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
    f2=np.sum(np.int8(f_e)*w,axis=1)
    f3=np.sum(np.int8(f)*w1,axis=1)

    risk=data_E.iloc[:,34]
    delay=data_E.iloc[:,39]
    m=[risk=='1']
    n=[delay=='1']
    a=np.array(np.where(np.array(m[0])==1))
    b=np.array(np.where(np.array(n[0])==1))
    tmp1 = [i for i in a[0] if i in b[0]]
    m=[risk=='2']
    n=[delay=='1']
    a=np.array(np.where(np.array(m[0])==1))
    b=np.array(np.where(np.array(n[0])==1))
    tmp2 = [i for i in a[0] if i in b[0]]
    m=[risk=='1']
    n=[delay=='2']
    a=np.array(np.where(np.array(m[0])==1))
    b=np.array(np.where(np.array(n[0])==1))
    tmp3 = [i for i in a[0] if i in b[0]]
    m=[risk=='2']
    n=[delay=='2']
    a=np.array(np.where(np.array(m[0])==1))
    b=np.array(np.where(np.array(n[0])==1))
    tmp4 = [i for i in a[0] if i in b[0]]
    tend[tmp4]
    f41=np.zeros(data_E.shape[0])
    f41[tmp1]=k[0,0]
    f41[tmp2]=k[1,0]
    f41[tmp3]=k[2,0]
    f41[tmp3]=k[3,0]
    f42=np.zeros(data_E.shape[0])
    f42[tmp1]=k[0,1]
    f42[tmp2]=k[1,1]
    f42[tmp3]=k[2,1]
    f42[tmp3]=k[3,1]
    f43=np.zeros(data_E.shape[0])
    f43[tmp1]=k[0,2]
    f43[tmp2]=k[1,2]
    f43[tmp3]=k[2,2]
    f43[tmp3]=k[3,2]
    f4=np.sum(np.vstack((f42,f41,f43)).T*w2,axis=1)
    time=np.float32(data_E.iloc[:,1:11])
    f5=np.sum(time*w_t,axis=1)
    return f1,f2,f3,f4,f5
def model_update(data_E):
    stress=np.int8(data_E.iloc[:,31])
    tired=np.int8(data_E.iloc[:,32])
    st=np.int8(data_E.iloc[:,31:33])
    choice3=data_E.iloc[:,37]
    choice4=data_E.iloc[:,38]
    tend=np.int8(choice3)+np.int8(choice4)
    poly_reg = PolynomialFeatures(degree=2)
    X_poly = poly_reg.fit_transform(st)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, tend)
    w0=pol_reg.coef_
    np.save('w0.npy',w0)
    data_p=np.zeros((data_E.shape[0]*10,13))
    data_clean=np.array(data_E)
    for i in range (0,data_E.shape[0]):
        data_p[i*10:i*10+10,0]=data_clean[i,0] #ID
        data_p[i*10:i*10+10,1]=data_clean[i,1:11] #time
        data_p[i*10:i*10+10,2]=data_clean[i,11:21] #choice
        data_p[i*10:i*10+10,3]=data_clean[i,21:31] #familarity
        data_p[i*10:i*10+10,4]=data_clean[i,31] #stress
        data_p[i*10:i*10+10,5]=data_clean[i,32] #tired
        data_p[i*10:i*10+10,6]=data_clean[i,33] #pay off
        data_p[i*10:i*10+10,7]=data_clean[i,34] #risk
        data_p[i*10:i*10+10,8]=data_clean[i,35] #bounce 4
        data_p[i*10:i*10+10,9]=data_clean[i,36] #bounce 3
        data_p[i*10:i*10+10,10]=data_clean[i,37] #choice 4 include 3?
        data_p[i*10:i*10+10,11]=data_clean[i,38] #choice 3
        data_p[i*10:i*10+10,12]=data_clean[i,39] #delay
        for i in range (1,5):
            y=data_p[np.where(data_p[:,3]==i),1]
            m=best_fit_distribution(y, bins=200, ax=None)
            np.save('m'+str(i)+'.npy',np.array(m[1]))
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
    lin_reg = LinearRegression()
    lin_reg.fit(np.int8(f_e), tend)
    w=lin_reg.coef_
    np.save('w.npy',w)
    lin_reg = LinearRegression()
    lin_reg.fit(np.int8(f), tend)
    w1=lin_reg.coef_
    np.save('w1.npy',w1)
    risk=data_E.iloc[:,34]
    delay=data_E.iloc[:,39]
    m=[risk=='1']
    n=[delay=='1']
    a=np.array(np.where(np.array(m[0])==1))
    b=np.array(np.where(np.array(n[0])==1))
    tmp1 = [i for i in a[0] if i in b[0]]
    if np.array(tem1).any():
    #print(np.mean(tend[tmp1]))
        k11=np.mean(tend[tmp1])
        #print(np.median(tend[tmp1]))
        k12=np.median(tend[tmp1])
        counts = np.bincount(tend[tmp1])
        #print(np.argmax(counts))
        k13=np.argmax(counts)
    else:
        k11=k[0,0]
        k12=k[0,1]
        k13=k[0,2]
    m=[risk=='2']
    n=[delay=='1']
    a=np.array(np.where(np.array(m[0])==1))
    b=np.array(np.where(np.array(n[0])==1))
    tmp2 = [i for i in a[0] if i in b[0]]
    if np.array(tem2).any():
    #tend[tmp2]
    #plt.hist(tend[tmp2])
    #print(np.mean(tend[tmp2]))
        k21=np.mean(tend[tmp2])
        #print(np.median(tend[tmp2]))
        k22=np.median(tend[tmp2])
        counts = np.bincount(tend[tmp2])
        #print(np.argmax(counts))
        k23=np.argmax(counts)
    else:
        k21=k[1,0]
        k22=k[1,1]
        k23=k[1,2]
    m=[risk=='1']
    n=[delay=='2']
    a=np.array(np.where(np.array(m[0])==1))
    b=np.array(np.where(np.array(n[0])==1))
    tmp3 = [i for i in a[0] if i in b[0]]
    if np.array(tem3).any():
    #tend[tmp3]
    #plt.hist(tend[tmp2])
    #print(np.mean(tend[tmp2]))
        k31=np.mean(tend[tmp2])
        #print(np.median(tend[tmp2]))
        k32=np.median(tend[tmp2])
        counts = np.bincount(tend[tmp2])
        #print(np.argmax(counts))
        k33=np.argmax(counts)
    else:
        k31=k[2,0]
        k32=k[2,1]
        k33=k[2,2]
    m=[risk=='2']
    n=[delay=='2']
    a=np.array(np.where(np.array(m[0])==1))
    b=np.array(np.where(np.array(n[0])==1))
    tmp4 = [i for i in a[0] if i in b[0]]
    if np.array(tem4).any():    
    #tend[tmp4]
    #plt.hist(tend[tmp2])
    #print(np.mean(tend[tmp2]))
        k41=np.mean(tend[tmp2])
        #print(np.median(tend[tmp2]))
        k42=np.median(tend[tmp2])
        counts = np.bincount(tend[tmp2])
        #print(np.argmax(counts))
        k43=np.argmax(counts)
    else:
        k41=k[3,0]
        k42=k[3,1]
        k43=k[3,2]
    k=np.zeros((4,3))
    k[0,0]=k11
    k[1,0]=k21
    k[2,0]=k31
    k[3,0]=k41
    k[0,1]=k12
    k[1,1]=k22
    k[2,1]=k32
    k[3,1]=k42
    k[0,2]=k13
    k[1,2]=k23
    k[2,2]=k33
    k[3,2]=k43
    np.save('k.npy',k)
    f41=np.zeros(data_E.shape[0])
    f41[tmp1]=k[0,0]
    f41[tmp2]=k[1,0]
    f41[tmp3]=k[2,0]
    f41[tmp3]=k[3,0]
    f42=np.zeros(data_E.shape[0])
    f42[tmp1]=k[0,1]
    f42[tmp2]=k[1,1]
    f42[tmp3]=k[2,1]
    f42[tmp3]=k[3,1]
    f43=np.zeros(data_E.shape[0])
    f43[tmp1]=k[0,2]
    f43[tmp2]=k[1,2]
    f43[tmp3]=k[2,2]
    f43[tmp3]=k[3,2]
    lin_reg = LinearRegression()
    lin_reg.fit(np.vstack((f42,f41,f43)).T, tend)
    w2=lin_reg.coef_
    np.save('w2.npy',w2)
    lin_reg = LinearRegression()
    time=np.float32(data_E.iloc[:,1:11])
    lin_reg.fit(time, tend)
    w_t=lin_reg.coef_
    np.save('w_t.npy',w_t)
    