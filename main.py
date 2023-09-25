#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:05:36 2020

@author: Urs Hofmann Elizondo
Main file for the analysis of species co-occurrences
"""

import numpy as np

import functions
import logging
import glob
import re
import multiprocessing
import csv

from tqdm import tqdm

def main():

    configuration_file = '/Users/urshofmannelizondo/Documents/PhD/Co-occurrence_analysis/Co_occurrence_analysis/parameters.yaml'

    my_config = functions.read_config_file(configuration_file)

    #Define the directory with the model output
    #The files are RData files
    directory = my_config.directory_R_data
    #get all data files with the annual compositon
    file_names = glob.glob(directory + my_config.R_data_file_root)
    
    #create an array with the same size as number of files, and 6 columns
    #The six columns will help identify the files according to the species (phyto/zoo),
    #baseline/future projection, algorithm used (GAM/GLM/RF/ANN), climate model used, and predictor pool
    list_files = np.zeros((len(file_names),6))
    list_files[:,0] = np.arange(len(file_names))
    
    models = []
    for i, file in enumerate(file_names):
    #populate matrix with values
        
        if my_config.first_group in file:
            list_files[i,1] = 1
   
        if my_config.reference_time_name in file:
            list_files[i,2] = 1

        for idx_algorithm, algorithm in enumerate(my_config.algorithm_names):
            
            if algorithm in file:
                list_files[i,3] = idx_algorithm + 1
                if list_files[i,2] == 0:
                    pattern = '2000_(.*)_' + algorithm + '_p'
                    substring = re.search(pattern, file)
                    models.append(substring.group(1))
        
        for idx_params, parameter in enumerate(my_config.predictor_set):
                     
            if parameter in file:
                list_files[i,5] = idx_params + 1

    all_models = np.unique(models)
    for i, file in enumerate(file_names):
        for j, model in enumerate(all_models):
            if model in file:
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
    clusters = None
    if my_config.directory_clusters is not None:
        clusters =  np.genfromtxt(my_config.directory_clusters, delimiter=',')

    #do the calculations in parallel
    jobs = []
    
    for my_list in range(len(list_of_values)):
        
        p = multiprocessing.Process(target=functions.calculate_scores, args=(list_files,file_names,list_of_values[my_list],clusters,all_models, my_config))

        jobs.append(p)
        p.start()
    
    for process in jobs:
            process.join()
            
            
#==============================================================================
# ANALYSIS PART
#==============================================================================
            
    #Define directory where data was stored above
    directory = my_config.directory_analysis
    
    #Choose the appropriate file names with Cluster or global
    if clusters is not None:
        file_names = glob.glob(directory+"Cluster*")
    else:
        file_names = glob.glob(directory+"Threshold*")
    
    algo_names = my_config.algorithm_names
    params = my_config.predictor_set

    models = []
    for file in file_names:
    #populate matrix with values
        pattern = "baseline_(.*)_ANN_p"
        substring = re.search(pattern, file)
        if substring is not None:
            models.append(substring.group(1))
    
    models = np.unique(models).tolist()   
    
    my_list = functions.find_baseline(file_names,models=models,algo_names=algo_names,params=params,clusters=clusters, configurations=my_config)   
    
    my_list = my_list[my_list[:,1] == 1,:]
    
    all_results = np.empty((3*len(my_list),8),dtype=float)
    all_results_text = []
    c = 0
    for i in tqdm(my_list, desc='Calculating statistics'):

        base_file = file_names[int(my_list[i,2])]
        future_file = file_names[int(my_list[i,0])]
        
        pattern = "Threshold_(.*)_Scores_baseline"
        thr = re.search(pattern, base_file).group(1)
        
        if clusters is not None:
            pattern = "Cluster_(.*)_Threshold"
            clust = re.search(pattern, base_file).group(1)
            logging.info(clust)
        
        algo = base_file[-10:-7]
        algo = re.sub('[!@#$_]', '', algo)
                        
        pattern = "baseline_(.*)_"+algo
        mod = re.search(pattern, base_file).group(1)
        p_val = base_file[-5]
         
        matrix_baseline = np.genfromtxt(base_file, delimiter=',')
        matrix_projection = np.genfromtxt(future_file, delimiter=',')
    
        
        logging.warning('All')
        loss_a,gain_a,const_a,never = functions.get_difference(matrix_baseline, matrix_projection)
        tot_a = loss_a + gain_a + const_a    
        logging.warning(np.array([loss_a,gain_a,const_a])/tot_a)
        
        logging.warning('phyto:')
        loss_p,gain_p,const_p,never = functions.get_difference(matrix_baseline[:338,:338], matrix_projection[:338,:338])
        tot_p = loss_p + gain_p + const_p
        logging.warning(np.array([loss_p,gain_p,const_p])/tot_p)
    
        logging.warning('zoo:')
        loss_z,gain_z,const_z,never = functions.get_difference(matrix_baseline[338:,338:], matrix_projection[338:,338:])
        tot_z = loss_z + gain_z + const_z
        logging.warning(np.array([loss_z,gain_z,const_z])/tot_z)
        
        logging.warning('phyto-zoo:')
        loss_pz,gain_pz,const_pz,never = functions.get_difference(matrix_baseline[:338,338:], matrix_projection[:338,338:])
        tot_pz = loss_pz + gain_pz + const_pz
        logging.warning(np.array([loss_pz,gain_pz,const_pz])/tot_pz)
        
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
    
    if clusters is not None:
        filename_results = my_config.directory_output + 'Cluster_All_results.csv'
        filename_results_description = my_config.directory_output + 'Cluster_All_results_description.csv'
    else:
        filename_results = my_config.directory_output + 'All_results.csv'
        filename_results_description = my_config.directory_output + 'All_results_description.csv'

    np.savetxt(filename_results, all_results, delimiter=',')  

    with open(filename_results_description,"w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(all_results_text)
    
    

'''
Main Function
'''
if __name__ in "__main__":
    logging.basicConfig(level=logging.WARNING)

    main()
    