#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:10:29 2020

@author: ursho
"""
import csv
import numpy as np
import glob
import re





def find_baseline(file_names,models,algo_names,params,clusters=None):
    
    flag_clusters = 0
    if clusters is not None:
        flag_clusters = 1
    
    list_files = np.zeros((len(file_names),3))
    list_files[:,0] = np.arange(len(file_names))
    threshold = np.arange(25,41,1)
    threshold2 = np.arange(10,26,1)
    
    if flag_clusters == 1:
        for cl in clusters:
            for thr in range(len(threshold)):
                for j in range(len(models)):
                    for k in range(len(algo_names)):
                        for l in range(len(params)):
                            
                            i = threshold[thr]
                            if algo_names[k] == 'RF':
                                i = threshold2[thr]
                            
                            future_name = 'Cluster_'+str(int(cl))+'_Threshold_'+str(i)+'_Scores_future_'+models[j]+'_'+algo_names[k]+'_'+params[l]+'.csv'
                            baseline_name = 'Cluster_'+str(int(cl))+'_Threshold_'+str(i)+'_Scores_baseline_'+models[j]+'_'+algo_names[k]+'_'+params[l]+'.csv'
                            
                            ind_future = [ii for ii in range(len(file_names)) if future_name in file_names[ii]]
                            ind_baseline = [ii for ii in range(len(file_names)) if baseline_name in file_names[ii]]
                            list_files[ind_future,1] = 1
                            list_files[ind_future,2] = ind_baseline
                            
    else:
        for thr in range(len(threshold)):
            for j in range(len(models)):
                for k in range(len(algo_names)):
                    for l in range(len(params)):
                        
                        i = threshold[thr]
                        if algo_names[k] == 'RF':
                            i = threshold2[thr]
                        
                        future_name = 'Threshold_'+str(i)+'_Scores_future_'+models[j]+'_'+algo_names[k]+'_'+params[l]+'.csv'
                        baseline_name = 'Threshold_'+str(i)+'_Scores_baseline_'+models[j]+'_'+algo_names[k]+'_'+params[l]+'.csv'
                        
                        ind_future = [ii for ii in range(len(file_names)) if future_name in file_names[ii]]
                        ind_baseline = [ii for ii in range(len(file_names)) if baseline_name in file_names[ii]]
                        list_files[ind_future,1] = 1
                        list_files[ind_future,2] = ind_baseline
        
  
    return list_files      



def get_difference(matrix_baseline, matrix_projection,threshold=75):
    #output [diff_matrix,loss,gain,never,always]
    #%This function uses the LLR scores between species in a baseline and a 
    #%projection, and a threshold to decide what values are significant. 
    #%Significant LLR values need to be positive.The threshold is given as a 
    #%percentile. If the threshold is a missing values then significant LLR 
    #%values are all those that are positive.
    n,m = matrix_baseline.shape
    vec = matrix_baseline[matrix_baseline > 0]

    Y = np.percentile(vec, threshold)

    baseline = np.where(matrix_baseline > Y,1,0)
    future = np.where(matrix_projection > Y,1,0)
    
    
    loss = len(np.argwhere((baseline == 1) & (future == 0)))
    gain = len(np.argwhere((baseline == 0) & (future == 1)))
    const = len(np.argwhere((baseline == 1) & (future == 1)))
    never = len(np.argwhere((baseline == 0) & (future == 0)))
    
    return [loss,gain,const,never]

def flag_significant_pairs(matrix_baseline, matrix_projection,threshold=75):
    #output [matrix_significant]
    #This function flags species pairs lost, gained, and remained with a 1,2, and 3, respectively
    #It uses the LLR scores between species in a baseline and a projection,
    #and a threshold to decide what values are significant.
    #Significant LLR values need to be positive. The threshold is given as a percentile.
    #If the threshold is a missing value, then significant LLR values are all those that are positive
    

    vec = matrix_baseline[matrix_baseline > 0]

    Y = np.percentile(vec, threshold)

    gained_pairs = np.where((matrix_baseline <= Y) & (matrix_projection > Y),1,0)
    
    return gained_pairs

    



