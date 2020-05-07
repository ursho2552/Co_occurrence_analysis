#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:57:48 2020

@author: ursho
"""
import numpy as np
import pyreadr




def Calculate_Scores(list_files,file_names,list_of_values,clusters,model_names,algo_names=['GAM','GLM','RF','ANN'],output_dir="/home/ursho/PhD/Projects/Communitiy_cooccurrence_Fabio/All_models/" ):
    
    #first step is to find the indeces of the files needed as depicted in list_of_values
    x1,x2,x3 = list_of_values
    f_phyto_base = np.where((list_files[:,1] == 1) & (list_files[:,2] == 1) & (list_files[:,3] == x3) &
                               (list_files[:,4] == 0) & (list_files[:,5] == x1))[0][0]
    
    f_phyto_future = np.where((list_files[:,1] == 1) & (list_files[:,2] == 0) & (list_files[:,3] == x3) &
                               (list_files[:,4] == x2) & (list_files[:,5] == x1))[0][0]
    
    f_zoo_base = np.where((list_files[:,1] == 0) & (list_files[:,2] == 1) & (list_files[:,3] == x3) &
                               (list_files[:,4] == 0) & (list_files[:,5] == x1))[0][0]  
    
    f_zoo_future = np.where((list_files[:,1] == 0) & (list_files[:,2] == 0) & (list_files[:,3] == x3) &
                               (list_files[:,4] == x2) & (list_files[:,5] == x1))[0][0]

    phyto_baseline = read_in_data(file_names[f_phyto_base])
    phyto_future = read_in_data(file_names[f_phyto_future])
    zoo_baseline = read_in_data(file_names[f_zoo_base])
    zoo_future = read_in_data(file_names[f_zoo_future])
    
    
    name_file = model_names[x2-1]+'_'+algo_names[x3-1]+'_p'+str(x1)+'.csv' 
    print(name_file)
    current_algo = algo_names[x3-1]
    
    
    #select matrix with highest number of cells

    if phyto_future.shape[0] > zoo_future.shape[0]:
        
        ID_matrix = phyto_future[:,[0,2,3]]
#        flag_phyto = 1;
    else:
        ID_matrix = zoo_future[:,[0,2,3]]
#        flag_phyto = 0;
        
   #transform all datasets to the same format
    N = max(phyto_future.shape[0],zoo_future.shape[0])
    M_phyto = phyto_future.shape[1]
    M_zoo = zoo_future.shape[1]
#    if flag_phyto == 1:
#        N = phyto_future;
#    else:
#        N = zoo_future;

    print('Homogenizing data...')
    
    phyto_baseline_extended = np.full((N,M_phyto),np.nan)
    phyto_future_extended = np.full((N,M_phyto),np.nan)
    zoo_baseline_extended = np.full((N,M_zoo),np.nan)
    zoo_future_extended = np.full((N,M_zoo),np.nan)
    
    flag_clusters = 0
    if clusters is not None:
        clusters_extended = np.full((N,4),np.nan)
        flag_clusters = 1
    
    for i in range(N):
        if i < phyto_baseline.shape[0]:
            ind = np.argwhere((ID_matrix[:,1] == phyto_baseline[i,2]) & (ID_matrix[:,2] == phyto_baseline[i,3]))
            if ind.shape[0]>0:
                phyto_baseline_extended[ind,:] = phyto_baseline[i,:]
        if i < phyto_future.shape[0]:
            ind = np.argwhere((ID_matrix[:,1] == phyto_future[i,2]) & (ID_matrix[:,2] == phyto_future[i,3]))
            if ind.shape[0]>0:
                phyto_future_extended[ind,:] = phyto_future[i,:]
        if i < zoo_baseline.shape[0]:
            ind = np.argwhere((ID_matrix[:,1] == zoo_baseline[i,2]) & (ID_matrix[:,2] == zoo_baseline[i,3]))
            if ind.shape[0]>0:
                zoo_baseline_extended[ind,:] = zoo_baseline[i,:]
        if i < zoo_future.shape[0]:
            ind = np.argwhere((ID_matrix[:,1] == zoo_future[i,2]) & (ID_matrix[:,2] == zoo_future[i,3]))
            if ind.shape[0]>0:
                zoo_future_extended[ind,:] = zoo_future[i,:]
        if clusters is not None:
            if i < clusters.shape[0]:
                ind = np.argwhere((ID_matrix[:,1] == clusters[i,2]) & (ID_matrix[:,2] == clusters[i,3]))
                if ind.shape[0]>0:
                    clusters_extended[ind,:] = clusters[i,[0,2,3,4]]
            
                

    #Clean data and only use values found in both baseline and future
    phyto_future_extended[np.isnan(phyto_baseline_extended[:,4]),4:] = np.nan
    phyto_baseline_extended[np.isnan(phyto_future_extended[:,4]),4:] = np.nan
    
    zoo_future_extended[np.isnan(zoo_baseline_extended[:,4]),4:]= np.nan
    zoo_baseline_extended[np.isnan(zoo_future_extended[:,4]),4:] = np.nan
    
    both_baseline = np.hstack((phyto_baseline_extended,zoo_baseline_extended[:,4:]))
    both_future = np.hstack((phyto_future_extended,zoo_future_extended[:,4:]))
     
    


#    file_names = glob.glob(output_dir+"Threshold_*.csv")
    threshold = np.arange(25,41,1)
    threshold_RF = np.arange(10,26,1)
    if flag_clusters == 0:
        #Gloabl view!
        for thr in range(len(threshold)):
            i = threshold[thr]
            if current_algo == 'RF':
                i = threshold_RF[thr]
                print('Threshold: {}'.format(i))
            #unindent part below for later
            filename_base = output_dir+'Global/Threshold_'+str(i)+'_Scores_baseline_'+name_file
            N_spec = both_baseline.shape[1]-4
            Global_Scores_both_baseline = np.full((N_spec,N_spec),np.nan)
            
            data = (both_baseline > i/100).astype(float)
            data[np.isnan(both_baseline)] = np.nan
            
            Global_Scores_both_baseline[:,:] = Confusion_matrix(data)
            np.savetxt(filename_base, Global_Scores_both_baseline, delimiter=',')
                
    
            
            filename_future = output_dir+'Global/Threshold_'+str(i)+'_Scores_future_'+name_file
            N_spec = both_future.shape[1]-4
            Global_Scores_both_future = np.full((N_spec,N_spec),np.nan)
            
            data = (both_future > i/100).astype(float)
            data[np.isnan(both_future)] = np.nan
            
            Global_Scores_both_future[:,:] = Confusion_matrix(data)
            np.savetxt(filename_future, Global_Scores_both_future, delimiter=',')
            
    elif flag_clusters == 1:
        #cluster view
        num_clust = np.unique(clusters_extended[~np.isnan(clusters_extended[:,-1]),-1])
        for thr in range(len(threshold)):
            i = threshold[thr]
            if current_algo == 'RF':
                i = threshold_RF[thr]
                
            for n in num_clust:
                filename_base = output_dir+'Clusters/Cluster_'+str(int(n))+'_Threshold_'+str(i)+'_Scores_baseline_'+name_file
                N_spec = both_baseline.shape[1]-4
                Global_Scores_both_baseline = np.full((N_spec,N_spec),np.nan)
                
                data = (both_baseline > i/100).astype(float)
                data[np.isnan(both_baseline)] = np.nan
                data = data[clusters_extended[:,-1]== n,:]
                
                Global_Scores_both_baseline[:,:] = Confusion_matrix(data)
                np.savetxt(filename_base, Global_Scores_both_baseline, delimiter=',')
                    
        
                
                filename_future = output_dir+'Clusters/Cluster_'+str(int(n))+'_Threshold_'+str(i)+'_Scores_future_'+name_file
                N_spec = both_future.shape[1]-4
                Global_Scores_both_future = np.full((N_spec,N_spec),np.nan)
                
                data = (both_future > i/100).astype(float)
                data[np.isnan(both_future)] = np.nan
                data = data[clusters_extended[:,-1]== n,:]
                
                
                Global_Scores_both_future[:,:] = Confusion_matrix(data)
                np.savetxt(filename_future, Global_Scores_both_future, delimiter=',')

    return


def read_in_data(data):
    result = pyreadr.read_r(data)
    items = list(result.items())
    
    tmp = result[items[0][0]]
    tmp['cell_id'] = np.nan
    tmp.insert(0, 'index', tmp.index)
    res = tmp.to_numpy(dtype=float,copy=True)
    print(res.shape)
    return res

def Confusion_matrix(data):
    # data contains the presence probability of all species in a given region/cluster
    # rows are the pixels available
    # columns are the different species
    rows, cols = data.shape
    
    print('Calculating co-occurrence scores...')
    Score = np.zeros((cols-4,cols-4))
    
    for j in range(4,cols-1):
        print(j,cols-1)

        sum_dat1 = np.sum(data[:,j]>0)
        Score[j-4,j+1-4:] = [Get_Dunning(data[:,j]+data[:,k],data[:,j]-data[:,k],sum_dat1) for k in range(j+1,cols)]

    return Score

def Get_Dunning(dat_pos,dat_neg,sum_dat1):

    dat_c1 = len(dat_pos[dat_pos == 2])
    dat_c2 = len(dat_neg[dat_neg == -1])
    dat_c3 = len(dat_neg[dat_neg == 1])
    dat_c4 = len(dat_pos[dat_pos == 0])
    

    check = np.array([[dat_c1,dat_c2],
                       [dat_c3,dat_c4]])

    N_tokens = np.sum(dat_pos >= 0)
    k_vec = np.reshape(check,(1, 4))

    res = Get_LLR(check,k_vec)
        
    #Add Evert et al 2008 to distinguish between positive and negative associations
    #Absolute value expresses significance

    Expected = sum_dat1*(dat_c1+dat_c2)/N_tokens

    if(check[0,0] < Expected):
        return -res
    return res
   

def Shannon(vec):
    #input has to be a vector       
    n = np.sum(vec)
    H = np.sum(vec/n * np.log(vec/n + (vec == 0)))
              
    return H

def Get_LLR(check,k_vec):
    
    return  2 * np.sum(k_vec) * (Shannon(k_vec) -
                       Shannon(np.sum(check,1)) - Shannon(np.sum(check,0)))


        




    