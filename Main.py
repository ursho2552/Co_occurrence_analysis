#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:05:36 2020

@author: Urs Hofmann Elizondo
Main file for the analysis of species co-occurrences
"""
from Scores import Calculate_Scores
from Analysis import find_baseline, get_difference
import numpy as np
import glob
import re
import multiprocessing

if __name__ in "__main__":
    #Define the directory with the model output
    #The files are RData files
    directory = '/net/kryo/work/fabioben/OVERSEE/data/tables_composition_ensemble_rcp85/Individual_projections/'
    #get all data files with the annual compositon
    file_names = glob.glob(directory+"table_ann_compo_*.Rdata")
    
    #create an array with the same size as number of files, and 6 columns
    #The six columns will help identify the files according to the species (phyto/zoo),
    #baseline/future projection, algorithm used (GAM/GLM/RF/ANN), climate model used, and predictor pool
    list_files = np.zeros((len(file_names),6))
    list_files[:,0] = np.arange(len(file_names))
    
    models = []
    for i in range(len(file_names)):
    #populate matrix with values
        s = file_names[i]
        
        if 'phyto' in file_names[i]:
            list_files[i,1] = 1
   
        if 'baseline' in file_names[i]:
            list_files[i,2] = 1
    
        if 'GAM' in file_names[i]:
            list_files[i,3] = 1
            if list_files[i,2] == 0:
                pattern = "2000_(.*)_GAM_p"
                substring = re.search(pattern, s)
                models.append(substring.group(1))
                
        elif 'GLM' in file_names[i]:
            list_files[i,3] = 2
            if list_files[i,2] == 0:
                pattern = "2000_(.*)_GLM_p"
                substring = re.search(pattern, s)
                models.append(substring.group(1))
                
        elif 'RF' in file_names[i]:
            list_files[i,3] = 3
            if list_files[i,2] == 0:
                pattern = "2000_(.*)_RF_p"
                substring = re.search(pattern, s)
                models.append(substring.group(1))
                
        elif 'ANN' in file_names[i]:
            list_files[i,3] = 4
            if list_files[i,2] == 0:
                pattern = "2000_(.*)_ANN_p"
                substring = re.search(pattern, s)
                models.append(substring.group(1))
            
        if 'p1' in file_names[i]:
            list_files[i,5] = 1
        elif 'p2' in file_names[i]:
            list_files[i,5] = 2
        elif 'p3' in file_names[i]:
            list_files[i,5] = 3
        elif 'p4' in file_names[i]:
            list_files[i,5] = 4
            
    all_models = np.unique(models)
    for i in range(len(file_names)):
        for j in range(len(all_models)):
            if all_models[j] in file_names[i]:
                list_files[i,4] = j+1
                 
    
    #=====================================================================
    # Now read in the data, clean it, merge it, and calculate the Scores
    #=====================================================================
    #Now that we have a unique code for each file in list_files, we can easily identify each future projection and the corresponding baseline
    #Create a list of the possible combinations
    
    list_of_values = []
    for i in range(1,max(np.unique(list_files[:,-1])).astype(int)+1):
        for j in range(1,max(np.unique(list_files[:,-2])).astype(int)+1):
            for k in range(1,max(np.unique(list_files[:,-3])).astype(int)+1):
                list_of_values.append([i,j,k])
                


    #read in the clusters
    clusters =  np.genfromtxt('/home/ursho/PhD/Projects/Communitiy_cooccurrence_Fabio/All_models/Table_clusters.csv', delimiter=',')
    
    #do the calculations in parallel
    #We performed the analysis in chunks of ten on different machines
    jobs = []
    
    for my_list in range(len(list_of_values)):
        
        p = multiprocessing.Process(target=Calculate_Scores, args=(list_files,file_names,list_of_values[my_list],clusters,all_models))
        jobs.append(p)
        p.start()
    
    for process in jobs:
            process.join()
            
            
#==============================================================================
# ANALYSIS PART
#==============================================================================
            
    #Define directory where data was stored above
    #Choose one either gloabl or cluster:
#    directory = '/home/ursho/PhD/Projects/Communitiy_cooccurrence_Fabio/All_models/Global/'
    directory = '/home/ursho/PhD/Projects/Communitiy_cooccurrence_Fabio/All_models/Clusters/'
    
    #Choose the appropriate file names with Cluster or global
#    file_names = glob.glob(directory+"Threshold*")
    file_names = glob.glob(directory+"Cluster*")
    
    models = []
    algo_names = ['GAM','GLM','RF','ANN']
    params = ['p1','p2','p3','p4']
    for i in range(len(file_names)):
    #populate matrix with values
        s = file_names[i]
        pattern = "baseline_(.*)_ANN_p"
        substring = re.search(pattern, s)
        if substring is not None:
            models.append(substring.group(1))
    
    models = np.unique(models)   
    
#    clusters = None
    clusters = np.arange(1,7)
    my_list = find_baseline(file_names,models=models,algo_names=algo_names,params=params,clusters=clusters)   
    
    my_list = my_list[my_list[:,1] == 1,:]
    
    all_results = np.empty((3*len(my_list),8),dtype=float)
    all_results_text = []
    c = 0;
    for i in range(len(my_list)):
        base_file = file_names[int(my_list[i,2])]
        future_file = file_names[int(my_list[i,0])]
        
        s = base_file
        pattern = "Threshold_(.*)_Scores_baseline"
        thr = re.search(pattern, s).group(1)
        
        if clusters is not None:
            pattern = "Cluster_(.*)_Threshold"
            clust = re.search(pattern,s).group(1)
            print(clust)
        
        algo = s[-10:-7]
        algo = re.sub('[!@#$_]', '', algo)
                        
        pattern = "baseline_(.*)_"+algo
        mod = re.search(pattern, s).group(1)
        p_val = s[-5]
         
        matrix_baseline = np.genfromtxt(base_file, delimiter=',')
        matrix_projection = np.genfromtxt(future_file, delimiter=',')
    
        #all
        print('All')
        loss_a,gain_a,const_a,never = get_difference(matrix_baseline, matrix_projection)
        tot_a = loss_a+gain_a+const_a    
        print(np.array([loss_a,gain_a,const_a])/tot_a)
        
        print('phyto:')
        loss_p,gain_p,const_p,never = get_difference(matrix_baseline[:338,:338], matrix_projection[:338,:338])
        tot_p = loss_p+gain_p+const_p
        print(np.array([loss_p,gain_p,const_p])/tot_p)
    
        print('zoo:')
        loss_z,gain_z,const_z,never = get_difference(matrix_baseline[338:,338:], matrix_projection[338:,338:])
        tot_z = loss_z+gain_z+const_z
        print(np.array([loss_z,gain_z,const_z])/tot_z)
        
        print('phyto-zoo:')
        loss_pz,gain_pz,const_pz,never = get_difference(matrix_baseline[:338,338:], matrix_projection[:338,338:])
        tot_pz = loss_pz+gain_pz+const_pz
        print(np.array([loss_pz,gain_pz,const_pz])/tot_pz)
        
    
        all_results[c,:] = [loss_a/tot_a,loss_p/tot_p,loss_z/tot_z,loss_pz/tot_pz,tot_a,tot_p,tot_z,tot_pz]
        if clusters is not None:
            all_results_text.append([str(clust),'Loss',str(thr),mod,algo,str(p_val)])
        else:
            all_results_text.append(['Loss',str(thr),mod,algo,str(p_val)])
        c = c + 1
        
        all_results[c,:] = [gain_a/tot_a,gain_p/tot_p,gain_z/tot_z,gain_pz/tot_pz,tot_a,tot_p,tot_z,tot_pz]
        if clusters is not None:
            all_results_text.append([str(clust),'Gain',str(thr),mod,algo,str(p_val)])
        else:
            all_results_text.append(['Gain',str(thr),mod,algo,str(p_val)])
        c = c + 1
        
        all_results[c,:] = [const_a/tot_a,const_p/tot_p,const_z/tot_z,const_pz/tot_pz,tot_a,tot_p,tot_z,tot_pz]
        if clusters is not None:
            all_results_text.append([str(clust),'Const',str(thr),mod,algo,str(p_val)])
        else:
            all_results_text.append(['Const',str(thr),mod,algo,str(p_val)])
        c = c + 1
        
        print(i,len(my_list))
    
        
        
    filename_results = '/home/ursho/PhD/Projects/Communitiy_cooccurrence_Fabio/All_models/Cluster_All_results.csv'
    np.savetxt(filename_results, all_results, delimiter=',')  
    
    filename_results = '/home/ursho/PhD/Projects/Communitiy_cooccurrence_Fabio/All_models/Cluster_All_results_description.csv'
    with open(filename_results,"w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(all_results_text)
    
    

