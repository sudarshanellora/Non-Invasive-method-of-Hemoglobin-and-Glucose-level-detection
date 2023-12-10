# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:10:11 2023

@author: sudar
"""

#!/usr/bin/env python
# coding: utf-8

# # Training

# In[13]:


import pandas as pd
from scipy.stats import skew,kurtosis,entropy,median_abs_deviation
from ast import literal_eval
import pandas as pd
import matplotlib.pyplot as plt 
from scipy import signal
from scipy.fftpack import rfft, irfft, fftfreq, fft,ifft
from scipy.signal import find_peaks, peak_prominences
from numpy import trapz
from scipy.stats import skew as find_skew
import numpy as np
from scipy.stats import skew
import math
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
def training():
    try:
        df=pd.read_csv(r"file3.csv")  #give your file path here
#         print("Count of PPG Signal : ",len(df))
#         print(df.head())
        def stat_features_of_ppg(df):
            def motion_removal(data):

                Fs = 100;                                                   #Sampling Frequency (Hz)
                Fn = Fs/2;                                                  # Nyquist Frequency (Hz)
                Ws = 0.5/Fn;                                                # Passband Frequency Vector (Normalised)
                Wp = 1.5/Fn;                                                # Stopband Frequency Vector (Normalised)
                Rp =   1;                                                   # Passband Ripple (dB)
                Rs = 150; 
                N, Wn = signal.ellipord(Wp,Ws,Rp,Rs)
                z,p,k = signal.ellip(N,Rp,Rs,Wn,'high',output='zpk')
                sos= signal.zpk2sos(z, p, k)
                y = signal.sosfiltfilt(sos, data)
                return y
            def plotting(ir):
                plt.figure(figsize=(16,4))
                plt.plot(ir,label='IR')
                plt.legend()
                plt.figure(figsize=(16,4))
                plt.plot(red,color="orange",label='RED',)
                plt.legend()
            def onecycle(ppg):
                length=len(ppg)
                peaks, _ = find_peaks(x, distance=15)
                min1=9999999
                min2=9999999
                for i in range(peaks[0],peaks[1]):
                    if ppg[i]<min1:
                        min1=i
                for i in range(peaks[1],peaks[2]):
                    if ppg[i]<min2:
                        min2=i
                if abs(peaks[2]-peaks[1])>22.5:
                    min2=peaks[0]+min1
            #     print(min1,min2)
                final=ppg[min1-1:min2]

                mean=np.mean(final)
                std=np.std(final)
                median = np.median(final)
                skewness=skew(final)
                kurt=kurtosis(final)
                p10,p25,p30,p50,p75,p80,p90 = np.nanpercentile(final,[10,25,30,50,75,80,90])
                IQR = p75-p25
                EN = entropy(final)
                mid_AD = median_abs_deviation(final)
                mean_AD = sum([abs(i-np.mean(final)) for i in final])/len(final)
                RMS =  np.sqrt(sum([i**2 for i in final])/len(final))
                spec_EN = entropy(final)/np.log2(len(final))
                #print([mean,std,skewness,kurt,p10,p25,p30,p50,p75,p80,p90])
                return [mean,median,std,kurt,skewness,p10,p25,p30,p50,p75,p80,p90,IQR,EN,mid_AD,mean_AD,RMS,spec_EN]   


            data=pd.DataFrame(columns=['ID','mean','median','std','kurt','skewness','p10','p25','p30','p50','p75','p80','p90','IQR','EN','mid_AD','mean_AD','RMS','spec_EN'])
            for i in range(len(df)):
                ppg=df.iloc[i].ir.strip('][').split(', ')
                ppg= list(map(int,ppg))
                data1 =  np.flip(motion_removal(ppg[11:]))
                x=data1[150:300]
                feature=onecycle(x)
                index=int(df.iloc[i].ID)
                #print(features[0])

                data=data.append({'ID':int(index),'mean':feature[0],'median':feature[1],'std':feature[2],'kurt':feature[3],'skewness':feature[4],'p10':feature[5],'p25':feature[6],'p30':feature[7],'p50':feature[8],'p75':feature[9],'p80':feature[10],'p90':feature[11],'IQR':feature[12],'EN':feature[13],'mid_AD':feature[14],'mean_AD':feature[15],'RMS':feature[16],'spec_EN':feature[17]},ignore_index=True)
                #for i in range(0,len(features)):
                    #print(features[i])
            data['ID']=data['ID'].astype(int)
            df_cd = pd.merge(df, data,how='inner',on='ID')
            return df_cd


    # Morpho

        def morph_features_of_ppg(df):
            def motion_removal(data):

                Fs = 100;                                                   #Sampling Frequency (Hz)
                Fn = Fs/2;                                                  # Nyquist Frequency (Hz)
                Ws = 0.5/Fn;                                                # Passband Frequency Vector (Normalised)
                Wp = 1.5/Fn;                                                # Stopband Frequency Vector (Normalised)
                Rp =   1;                                                   # Passband Ripple (dB)
                Rs = 150; 
                N, Wn = signal.ellipord(Wp,Ws,Rp,Rs)
                z,p,k = signal.ellip(N,Rp,Rs,Wn,'high',output='zpk')
                sos= signal.zpk2sos(z, p, k)
                y = signal.sosfiltfilt(sos, data)
                return y
            def morphological_features(cycle):
                try:

                        sample_rate = 100

                        peaks,_ = find_peaks(cycle)
        #                 print(peaks)
                        if(len(peaks)!=0):
                            p0=peaks[0]
                        else:
                            peak=float('-inf')
                            for i in range(len(cycle)):
                                if cycle[i]>peak:
                                    peak=i
                            p0=peak
        #                 print(p0)
                        systolic_peak = int(p0)
                        dicortic_peak = int((len(cycle)+p0)/2)
                        tpi = (len(cycle)-1)/sample_rate
                        t2=dicortic_peak/sample_rate
                        t1=systolic_peak/sample_rate
        #                 print([cycle[systolic_peak],cycle[dicortic_peak],np.trapz(cycle[dicortic_peak:])/np.trapz(cycle[:dicortic_peak]),
        #                        cycle[dicortic_peak]/cycle[systolic_peak],dicortic_peak/sample_rate,systolic_peak/sample_rate,t2-t1,
        #                        t1/tpi,t2/tpi,cycle[systolic_peak]/t1,cycle[dicortic_peak]/t2,cycle[systolic_peak]/(tpi-t1),cycle[dicortic_peak]/(tpi-t2),
        #                        sum([abs(i)**2 for i in cycle])])
                #         print("hello")
                        return [cycle[systolic_peak],
                               cycle[dicortic_peak],
                               np.trapz(cycle[dicortic_peak:])/np.trapz(cycle[:dicortic_peak]),
                               cycle[dicortic_peak]/cycle[systolic_peak],
                               dicortic_peak/sample_rate,
                               systolic_peak/sample_rate,
                               t2-t1,
                               t1/tpi,
                               t2/tpi,
                               cycle[systolic_peak]/t1,
                               cycle[dicortic_peak]/t2,
                               cycle[systolic_peak]/(tpi-t1),
                               cycle[dicortic_peak]/(tpi-t2),
                               sum([abs(i)**2 for i in cycle])]

                except:
                    print("could not caluculate morphology feature for this ppg signal pk-pk")

                    raise Exception("could not caluculate for this cycle")
            def onecycle(ppg):
                length=len(ppg)
                peaks, _ = find_peaks(x, distance=15)
                min1=9999999
                min2=9999999
                for i in range(peaks[0],peaks[1]):
                    if ppg[i]<min1:
                        min1=i
                for i in range(peaks[1],peaks[2]):
                    if ppg[i]<min2:
                        min2=i
                if abs(peaks[2]-peaks[1])>22.5:
                    min2=peaks[0]+min1
            #     print(min1,min2)
                return ppg[min1-1:min2]

            data=pd.DataFrame(columns=["ID","morph1","morph2","morph3","morph4","morph5","morph6","morph7","morph8","morph9","morph10","morph11","morph12","morph13","morph14"])
            for i in range(len(df)):
                ppg=df.iloc[i].ir.strip('][').split(', ')
                ppg= list(map(int,ppg))
                data1 =  np.flip(motion_removal(ppg[11:]))
                x=data1[150:300]
                one_cycle=onecycle(x)
                mf=morphological_features(one_cycle)
                index=int(df.iloc[i].ID)
                data.loc[len(data)]=[int(index),mf[0],mf[1],mf[2],mf[3],mf[4],mf[5],mf[6],mf[7],mf[8],mf[9],mf[10],mf[11],mf[12],mf[13]]
            data['ID']=data['ID'].astype(int)
            return data
        statistical_features = stat_features_of_ppg(df)
        morph_features=morph_features_of_ppg(df) 
        all_features=pd.merge(statistical_features, morph_features,how='inner',on='ID')
        features_df=all_features[['mean','median','std','kurt','skewness','p10','p25','p30','p50','p75','p80','p90','IQR','mid_AD','mean_AD','RMS',"morph1","morph2","morph3","morph4","morph5","morph6","morph7","morph8","morph9","morph10","morph11","morph12","morph13","morph14",'Glucose level mg/dL','Hemoglobin level g/dL']]
        corr = features_df.iloc[:,:-2].corr()
        def correlation(dataset, threshold):
            col_corr = set()  # Set of all the names of correlated columns
            corr_matrix = dataset.corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if (corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                        colname = corr_matrix.columns[i]  # getting the name of column
                        col_corr.add(colname)
            return col_corr
        corr_features = correlation(features_df.loc[:,:], 0.85)
        features_df.drop(corr_features,axis=1,inplace=True)
        
#         Training
        features_df['Hemoglobin level g/dL'] =all_features['Hemoglobin level g/dL']
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        scaler = StandardScaler()
        X=scaler.fit_transform(features_df.iloc[:,:-2])
        #y=scaler.fit_transform(np.array(features_df['Hemoglobin level g/dL'].values).reshape(-1,1))

        y=scaler.fit_transform(np.array(features_df['Hemoglobin level g/dL'].values).reshape(-1,1))
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        pca=PCA(n_components=14)
        X = imputer.fit_transform(X)
        pca.fit(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1,random_state =123)
        
#         SVR

        svr = SVR(kernel = 'rbf')
        svr.fit(X_train, y_train)
        filename0 = 'svr.sav'
        joblib.dump(svr, filename0)
        
        
#         KNN
        k = 8 # Number of nearest neighbors to consider
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        filename1='knn_hb.sav'
        joblib.dump(knn, filename1)
        
        
        
#         Glucose Training
        X=scaler.fit_transform(features_df.iloc[:,:-2])
        #y=scaler.fit_transform(np.array(features_df['Hemoglobin level g/dL'].values).reshape(-1,1))

        y=scaler.fit_transform(np.array(features_df['Glucose level mg/dL'].values).reshape(-1,1))
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        pca=PCA(n_components=14)
        X = imputer.fit_transform(X)
        pca.fit(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1,random_state =123)
        
#         KNN GLUCOSE
        k = 8 # Number of nearest neighbors to consider
        knn_glucose = KNeighborsRegressor(n_neighbors=k)
        knn_glucose.fit(X_train, y_train)
        filename2='knn_glucose.sav'
        joblib.dump(knn_glucose,filename2)
        
        print("Training completed")
        






    
    
    
    except Exception as e:
        print("Oopsie! Unable to open file.\nMake sure the file path is correct!")
    


# In[14]:


#training()


# # Hemoglobin Prediction

# In[15]:


import matplotlib.pyplot as plt
def plot():
    df =  pd.read_csv(r'C:\Users\sudar\major\data.csv',converters={'ir': literal_eval,'red': literal_eval})
    df.rename(columns = {'id':'ID'}, inplace = True)
    def motion_removal(data):

            Fs = 100;                                                   #Sampling Frequency (Hz)
            Fn = Fs/2;                                                  # Nyquist Frequency (Hz)
            Ws = 0.5/Fn;                                                # Passband Frequency Vector (Normalised)
            Wp = 1.5/Fn;                                                # Stopband Frequency Vector (Normalised)
            Rp =   1;                                                   # Passband Ripple (dB)
            Rs = 150; 
            N, Wn = signal.ellipord(Wp,Ws,Rp,Rs)
            z,p,k = signal.ellip(N,Rp,Rs,Wn,'high',output='zpk')
            sos= signal.zpk2sos(z, p, k)
            y = signal.sosfiltfilt(sos, data)
            return y
    plt.figure(figsize=(16,4))
    data1=motion_removal(df['ir'][0][11:])
    plt.plot(data1)
    


# In[16]:

7
plot()


# In[17]:


import pandas as pd
import csv
import joblib
# df = pd.merge(data,data1,on='Unique Ref. ID')
def real_time_prediction():
    df =  pd.read_csv(r'C:\Users\sudar\major\data.csv',converters={'ir': literal_eval,'red': literal_eval})
    df.rename(columns = {'id':'ID'}, inplace = True)
#     def motion_removal(data):

#             Fs = 100;                                                   #Sampling Frequency (Hz)
#             Fn = Fs/2;                                                  # Nyquist Frequency (Hz)
#             Ws = 0.5/Fn;                                                # Passband Frequency Vector (Normalised)
#             Wp = 1.5/Fn;                                                # Stopband Frequency Vector (Normalised)
#             Rp =   1;                                                   # Passband Ripple (dB)
#             Rs = 150; 
#             N, Wn = signal.ellipord(Wp,Ws,Rp,Rs)
#             z,p,k = signal.ellip(N,Rp,Rs,Wn,'high',output='zpk')
#             sos= signal.zpk2sos(z, p, k)
#             y = signal.sosfiltfilt(sos, data)
#             return y
#     plt.figure(figsize=(16,4))
#     data1=motion_removal(df['ir'][0][11:])
#     plt.plot(data1)
    
#     c=input("Is the PPG signal correct? [Y/N]").lower()
#     if(c=="y"):
    
    
    
    def all_data():
        df=pd.read_csv(r"file3.csv")  #give your file path here
#         print("Count of PPG Signal : ",len(df))
#         print(df.head())
        def stat_features_of_ppg(df):
            def motion_removal(data):

                Fs = 100;                                                   #Sampling Frequency (Hz)
                Fn = Fs/2;                                                  # Nyquist Frequency (Hz)
                Ws = 0.5/Fn;                                                # Passband Frequency Vector (Normalised)
                Wp = 1.5/Fn;                                                # Stopband Frequency Vector (Normalised)
                Rp =   1;                                                   # Passband Ripple (dB)
                Rs = 150; 
                N, Wn = signal.ellipord(Wp,Ws,Rp,Rs)
                z,p,k = signal.ellip(N,Rp,Rs,Wn,'high',output='zpk')
                sos= signal.zpk2sos(z, p, k)
                y = signal.sosfiltfilt(sos, data)
                return y
            def plotting(ir):
                plt.figure(figsize=(16,4))
                plt.plot(ir,label='IR')
                plt.legend()
                plt.figure(figsize=(16,4))
                plt.plot(red,color="orange",label='RED',)
                plt.legend()
            def onecycle(ppg):
                length=len(ppg)
                peaks, _ = find_peaks(x, distance=15)
                min1=9999999
                min2=9999999
                for i in range(peaks[0],peaks[1]):
                    if ppg[i]<min1:
                        min1=i
                for i in range(peaks[1],peaks[2]):
                    if ppg[i]<min2:
                        min2=i
                if abs(peaks[2]-peaks[1])>22.5:
                    min2=peaks[0]+min1
            #     print(min1,min2)
                final=ppg[min1-1:min2]

                mean=np.mean(final)
                std=np.std(final)
                median = np.median(final)
                skewness=skew(final)
                kurt=kurtosis(final)
                p10,p25,p30,p50,p75,p80,p90 = np.nanpercentile(final,[10,25,30,50,75,80,90])
                IQR = p75-p25
                EN = entropy(final)
                mid_AD = median_abs_deviation(final)
                mean_AD = sum([abs(i-np.mean(final)) for i in final])/len(final)
                RMS =  np.sqrt(sum([i**2 for i in final])/len(final))
                spec_EN = entropy(final)/np.log2(len(final))
                #print([mean,std,skewness,kurt,p10,p25,p30,p50,p75,p80,p90])
                return [mean,median,std,kurt,skewness,p10,p25,p30,p50,p75,p80,p90,IQR,EN,mid_AD,mean_AD,RMS,spec_EN]   


            data=pd.DataFrame(columns=['ID','mean','median','std','kurt','skewness','p10','p25','p30','p50','p75','p80','p90','IQR','EN','mid_AD','mean_AD','RMS','spec_EN'])
            for i in range(len(df)):
                ppg=df.iloc[i].ir.strip('][').split(', ')
                ppg= list(map(int,ppg))
                data1 =  np.flip(motion_removal(ppg[11:]))
                x=data1[150:300]
                feature=onecycle(x)
                index=int(df.iloc[i].ID)
                #print(features[0])

                data=data.append({'ID':int(index),'mean':feature[0],'median':feature[1],'std':feature[2],'kurt':feature[3],'skewness':feature[4],'p10':feature[5],'p25':feature[6],'p30':feature[7],'p50':feature[8],'p75':feature[9],'p80':feature[10],'p90':feature[11],'IQR':feature[12],'EN':feature[13],'mid_AD':feature[14],'mean_AD':feature[15],'RMS':feature[16],'spec_EN':feature[17]},ignore_index=True)
                #for i in range(0,len(features)):
                    #print(features[i])
            data['ID']=data['ID'].astype(int)
            df_cd = pd.merge(df, data,how='inner',on='ID')
            return df_cd


    # Morpho

        def morph_features_of_ppg(df):
            def motion_removal(data):

                Fs = 100;                                                   #Sampling Frequency (Hz)
                Fn = Fs/2;                                                  # Nyquist Frequency (Hz)
                Ws = 0.5/Fn;                                                # Passband Frequency Vector (Normalised)
                Wp = 1.5/Fn;                                                # Stopband Frequency Vector (Normalised)
                Rp =   1;                                                   # Passband Ripple (dB)
                Rs = 150; 
                N, Wn = signal.ellipord(Wp,Ws,Rp,Rs)
                z,p,k = signal.ellip(N,Rp,Rs,Wn,'high',output='zpk')
                sos= signal.zpk2sos(z, p, k)
                y = signal.sosfiltfilt(sos, data)
                return y
            def morphological_features(cycle):
                try:

                        sample_rate = 100

                        peaks,_ = find_peaks(cycle)
        #                 print(peaks)
                        if(len(peaks)!=0):
                            p0=peaks[0]
                        else:
                            peak=float('-inf')
                            for i in range(len(cycle)):
                                if cycle[i]>peak:
                                    peak=i
                            p0=peak
        #                 print(p0)
                        systolic_peak = int(p0)
                        dicortic_peak = int((len(cycle)+p0)/2)
                        tpi = (len(cycle)-1)/sample_rate
                        t2=dicortic_peak/sample_rate
                        t1=systolic_peak/sample_rate
        #                 print([cycle[systolic_peak],cycle[dicortic_peak],np.trapz(cycle[dicortic_peak:])/np.trapz(cycle[:dicortic_peak]),
        #                        cycle[dicortic_peak]/cycle[systolic_peak],dicortic_peak/sample_rate,systolic_peak/sample_rate,t2-t1,
        #                        t1/tpi,t2/tpi,cycle[systolic_peak]/t1,cycle[dicortic_peak]/t2,cycle[systolic_peak]/(tpi-t1),cycle[dicortic_peak]/(tpi-t2),
        #                        sum([abs(i)**2 for i in cycle])])
                #         print("hello")
                        return [cycle[systolic_peak],
                               cycle[dicortic_peak],
                               np.trapz(cycle[dicortic_peak:])/np.trapz(cycle[:dicortic_peak]),
                               cycle[dicortic_peak]/cycle[systolic_peak],
                               dicortic_peak/sample_rate,
                               systolic_peak/sample_rate,
                               t2-t1,
                               t1/tpi,
                               t2/tpi,
                               cycle[systolic_peak]/t1,
                               cycle[dicortic_peak]/t2,
                               cycle[systolic_peak]/(tpi-t1),
                               cycle[dicortic_peak]/(tpi-t2),
                               sum([abs(i)**2 for i in cycle])]

                except:
                    print("could not caluculate morphology feature for this ppg signal pk-pk")

                    raise Exception("could not caluculate for this cycle")
            def onecycle(ppg):
                length=len(ppg)
                peaks, _ = find_peaks(x, distance=15)
                min1=9999999
                min2=9999999
                for i in range(peaks[0],peaks[1]):
                    if ppg[i]<min1:
                        min1=i
                for i in range(peaks[1],peaks[2]):
                    if ppg[i]<min2:
                        min2=i
                if abs(peaks[2]-peaks[1])>22.5:
                    min2=peaks[0]+min1
            #     print(min1,min2)
                return ppg[min1-1:min2]

            data=pd.DataFrame(columns=["ID","morph1","morph2","morph3","morph4","morph5","morph6","morph7","morph8","morph9","morph10","morph11","morph12","morph13","morph14"])
            for i in range(len(df)):
                ppg=df.iloc[i].ir.strip('][').split(', ')
                ppg= list(map(int,ppg))
                data1 =  np.flip(motion_removal(ppg[11:]))
                x=data1[150:300]
                one_cycle=onecycle(x)
                mf=morphological_features(one_cycle)
                index=int(df.iloc[i].ID)
                data.loc[len(data)]=[int(index),mf[0],mf[1],mf[2],mf[3],mf[4],mf[5],mf[6],mf[7],mf[8],mf[9],mf[10],mf[11],mf[12],mf[13]]
            data['ID']=data['ID'].astype(int)
            return data
        statistical_features = stat_features_of_ppg(df)
        morph_features=morph_features_of_ppg(df) 
        all_features=pd.merge(statistical_features, morph_features,how='inner',on='ID')
        features_df=all_features[['mean','median','std','kurt','skewness','p10','p25','p30','p50','p75','p80','p90','IQR','mid_AD','mean_AD','RMS',"morph1","morph2","morph3","morph4","morph5","morph6","morph7","morph8","morph9","morph10","morph11","morph12","morph13","morph14",'Glucose level mg/dL','Hemoglobin level g/dL']]
        corr = features_df.iloc[:,:-2].corr()
        def correlation(dataset, threshold):
            col_corr = set()  # Set of all the names of correlated columns
            corr_matrix = dataset.corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if (corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                        colname = corr_matrix.columns[i]  # getting the name of column
                        col_corr.add(colname)
            return col_corr
        corr_features = correlation(features_df.loc[:,:], 0.85)
        features_df.drop(corr_features,axis=1,inplace=True)

#         Training
        features_df['Hemoglobin level g/dL'] =all_features['Hemoglobin level g/dL']
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return features_df

    def stat_features_of_ppg(df):
        def motion_removal(data):

            Fs = 100;                                                   #Sampling Frequency (Hz)
            Fn = Fs/2;                                                  # Nyquist Frequency (Hz)
            Ws = 0.5/Fn;                                                # Passband Frequency Vector (Normalised)
            Wp = 1.5/Fn;                                                # Stopband Frequency Vector (Normalised)
            Rp =   1;                                                   # Passband Ripple (dB)
            Rs = 150; 
            N, Wn = signal.ellipord(Wp,Ws,Rp,Rs)
            z,p,k = signal.ellip(N,Rp,Rs,Wn,'high',output='zpk')
            sos= signal.zpk2sos(z, p, k)
            y = signal.sosfiltfilt(sos, data)
            return y
        def plotting(ir):
            plt.figure(figsize=(16,4))
            plt.plot(ir,label='IR')
            plt.legend()
            plt.figure(figsize=(16,4))
            plt.plot(red,color="orange",label='RED',)
            plt.legend()
        def onecycle(ppg):
            length=len(ppg)
            peaks, _ = find_peaks(x, distance=15)
            min1=9999999
            min2=9999999
            for i in range(peaks[0],peaks[1]):
                if ppg[i]<min1:
                    min1=i
            for i in range(peaks[1],peaks[2]):
                if ppg[i]<min2:
                    min2=i
            if abs(peaks[2]-peaks[1])>22.5:
                min2=peaks[0]+min1
        #     print(min1,min2)
            final=ppg[min1-1:min2]

            mean=np.mean(final)
            std=np.std(final)
            median = np.median(final)
            skewness=skew(final)
            kurt=kurtosis(final)
            p10,p25,p30,p50,p75,p80,p90 = np.nanpercentile(final,[10,25,30,50,75,80,90])
            IQR = p75-p25
            EN = entropy(final)
            mid_AD = median_abs_deviation(final)
            mean_AD = sum([abs(i-np.mean(final)) for i in final])/len(final)
            RMS =  np.sqrt(sum([i**2 for i in final])/len(final))
            spec_EN = entropy(final)/np.log2(len(final))
            #print([mean,std,skewness,kurt,p10,p25,p30,p50,p75,p80,p90])
            return [mean,median,std,kurt,skewness,p10,p25,p30,p50,p75,p80,p90,IQR,EN,mid_AD,mean_AD,RMS,spec_EN]   


        data=pd.DataFrame(columns=['ID','mean','median','std','kurt','skewness','p10','p25','p30','p50','p75','p80','p90','IQR','EN','mid_AD','mean_AD','RMS','spec_EN'])
        for i in range(len(df)):
            ppg=df.iloc[i].ir#.strip('][').split(', ')
            ppg= list(map(int,ppg))
            data1 =  np.flip(motion_removal(ppg[11:]))
            x=data1[150:300]
            feature=onecycle(x)
            index=int(df.iloc[i].ID)
            #print(features[0])

            data=data.append({'ID':int(index),'mean':feature[0],'median':feature[1],'std':feature[2],'kurt':feature[3],'skewness':feature[4],'p10':feature[5],'p25':feature[6],'p30':feature[7],'p50':feature[8],'p75':feature[9],'p80':feature[10],'p90':feature[11],'IQR':feature[12],'EN':feature[13],'mid_AD':feature[14],'mean_AD':feature[15],'RMS':feature[16],'spec_EN':feature[17]},ignore_index=True)
            #for i in range(0,len(features)):
                #print(features[i])
        data['ID']=data['ID'].astype(int)
        df_cd = pd.merge(df, data,how='inner',on='ID')
        return df_cd
    def morph_features_of_ppg(df):
        def motion_removal(data):

            Fs = 100;                                                   #Sampling Frequency (Hz)
            Fn = Fs/2;                                                  # Nyquist Frequency (Hz)
            Ws = 0.5/Fn;                                                # Passband Frequency Vector (Normalised)
            Wp = 1.5/Fn;                                                # Stopband Frequency Vector (Normalised)
            Rp =   1;                                                   # Passband Ripple (dB)
            Rs = 150; 
            N, Wn = signal.ellipord(Wp,Ws,Rp,Rs)
            z,p,k = signal.ellip(N,Rp,Rs,Wn,'high',output='zpk')
            sos= signal.zpk2sos(z, p, k)
            y = signal.sosfiltfilt(sos, data)
            return y
        def morphological_features(cycle):
            try:

                    sample_rate = 100

                    peaks,_ = find_peaks(cycle)
    #                 print(peaks)
                    if(len(peaks)!=0):
                        p0=peaks[0]
                    else:
                        peak=float('-inf')
                        for i in range(len(cycle)):
                            if cycle[i]>peak:
                                peak=i
                        p0=peak
#                     print(p0)
                    systolic_peak = int(p0)
                    dicortic_peak = int((len(cycle)+p0)/2)
                    tpi = (len(cycle)-1)/sample_rate
                    t2=dicortic_peak/sample_rate
                    t1=systolic_peak/sample_rate
#                     print([cycle[systolic_peak],cycle[dicortic_peak],np.trapz(cycle[dicortic_peak:])/np.trapz(cycle[:dicortic_peak]),
#                            cycle[dicortic_peak]/cycle[systolic_peak],dicortic_peak/sample_rate,systolic_peak/sample_rate,t2-t1,
#                            t1/tpi,t2/tpi,cycle[systolic_peak]/t1,cycle[dicortic_peak]/t2,cycle[systolic_peak]/(tpi-t1),cycle[dicortic_peak]/(tpi-t2),
#                            sum([abs(i)**2 for i in cycle])])
            #         print("hello")
                    return [cycle[systolic_peak],
                           cycle[dicortic_peak],
                           np.trapz(cycle[dicortic_peak:])/np.trapz(cycle[:dicortic_peak]),
                           cycle[dicortic_peak]/cycle[systolic_peak],
                           dicortic_peak/sample_rate,
                           systolic_peak/sample_rate,
                           t2-t1,
                           t1/tpi,
                           t2/tpi,
                           cycle[systolic_peak]/t1,
                           cycle[dicortic_peak]/t2,
                           cycle[systolic_peak]/(tpi-t1),
                           cycle[dicortic_peak]/(tpi-t2),
                           sum([abs(i)**2 for i in cycle])]

            except:
                print("could not caluculate morphology feature for this ppg signal pk-pk")

                raise Exception("could not caluculate for this cycle")
        def onecycle(ppg):
            length=len(ppg)
            peaks, _ = find_peaks(x, distance=15)
            min1=9999999
            min2=9999999
            for i in range(peaks[0],peaks[1]):
                if ppg[i]<min1:
                    min1=i
            for i in range(peaks[1],peaks[2]):
                if ppg[i]<min2:
                    min2=i
            if abs(peaks[2]-peaks[1])>22.5:
                min2=peaks[0]+min1
        #     print(min1,min2)
            return ppg[min1-1:min2]

        data=pd.DataFrame(columns=["ID","morph1","morph2","morph3","morph4","morph5","morph6","morph7","morph8","morph9","morph10","morph11","morph12","morph13","morph14"])
        for i in range(len(df)):
            ppg=df.iloc[i].ir#.strip('][').split(', ')
            ppg= list(map(int,ppg))
            data1 =  np.flip(motion_removal(ppg[11:]))
            x=data1[150:300]
            one_cycle=onecycle(x)
            mf=morphological_features(one_cycle)
            index=int(df.iloc[i].ID)
            data.loc[len(data)]=[int(index),mf[0],mf[1],mf[2],mf[3],mf[4],mf[5],mf[6],mf[7],mf[8],mf[9],mf[10],mf[11],mf[12],mf[13]]
        data['ID']=data['ID'].astype(int)
        return data
    morph_features=morph_features_of_ppg(df)
    statistical_features = stat_features_of_ppg(df)
    all_features=pd.merge(statistical_features, morph_features,how='inner',on='ID')
    feat = all_features[['mean', 'median', 'std', 'kurt', 'skewness', 'p10', 'p75', 'morph3',
           'morph4', 'morph5', 'morph6', 'morph8', 'morph9', 'morph12']]
    scaler=StandardScaler()
    scale_fit = scaler.fit(all_data().iloc[:,:-2]) #save the mean and std. dev computed for your data.
    scaled_data = scale_fit.transform(feat.iloc[:,:]) 
#     predicted_output=model.predict()

    filename1 = 'knn.sav'
    #joblib.dump(regressor, filename)
    loaded_model_hb = joblib.load(filename1)
    predicted_output_hb=loaded_model_hb.predict(scaled_data)
    output_hb=scaler.fit_transform(np.array(all_data()['Hemoglobin level g/dL'].values).reshape(-1,1))
    output_hb=np.append(output_hb,predicted_output_hb)
    hb=round(scaler.inverse_transform(output_hb.reshape(-1,1))[-1][0],2)

    filename2 = 'knn_glucose.sav'
    #joblib.dump(regressor, filename)
    loaded_model_gc = joblib.load(filename2)
    predicted_output_gc=loaded_model_gc.predict(scaled_data)
    output_gc=scaler.fit_transform(np.array(all_data()['Glucose level mg/dL'].values).reshape(-1,1))
    output_gc=np.append(output_gc,predicted_output_gc)
    gc=round(scaler.inverse_transform(output_gc.reshape(-1,1))[-1][0],2)

    # output[0]=predicted_output[0]
#     print(output)
    print(f"Your Hemoglobin level is : {hb} g/dL \n Your Glucose level is : {gc} mg/dL")


# In[18]:



#real_time_prediction()


# In[11]:





# In[ ]:




