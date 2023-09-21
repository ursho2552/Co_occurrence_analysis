#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:57:48 2020

@author: ursho
"""
from typing import Tuple
from tqdm import tqdm
import numpy as np
import pyreadr
import logging


def calculate_scores(list_files: list[int], file_names: list[str], list_of_values: list[int], 
                    clusters: np.ndarray, model_names: list[str], configurations: Configuration_parameters) -> None:
    
    algo_names = configurations.algorithm_names
    output_dir = configurations.directory_output
    #first step is to find the indeces of the files needed as depicted in list_of_values
    x1,x2,x3 = list_of_values

    # 1: zoo or phyto | 2: future or baseline | 3: Algorithm | 4: Models used | 5: parameters (1--4)
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
     
    name_file = f'{model_names[x2-1]}_{algo_names[x3-1]}_p{str(x1)}.csv'
    logging.info(name_file)

    homogenized_baseline, homogenized_future, homogenized_clusters = homogenize_data(phyto_baseline, zoo_baseline, phyto_future, zoo_future, clusters=clusters )
    
    start_threshold = configurations.threshold_start[x3-1]
    end_threshold = configurations.threshold_end[x3-1]
    threshold_algo = np.arange(start_threshold, end_threshold, 1)

    for thr in tqdm(threshold_algo), desc='Getting scores for different threshholds'):

        presence_baseline, presence_future = get_presence_absence(homogenized_baseline, homogenized_future, thr)

        # Check if clusters are used or the global view
        if homogenized_clusters is not None:
            #separate presence data into clusters and then calculate the confusion matrix
            number_clusters = np.unique(homogenized_clusters[~np.isnan(homogenized_clusters[:,-1]), -1])

            for cluster in range(number_clusters):

                copy_presence_baseline = presence_baseline.copy()
                copy_presence_future = presence_future.copy()

                cluster_baseline = copy_presence_baseline[homogenized_clusters[:,-1] == cluster, :]
                cluster_future = copy_presence_future[homogenized_clusters[:,-1] == cluster, :]

                scores_baseline = confusion_matrix(cluster_baseline)
                scores_future = confusion_matrix(cluster_future)

                save_scores(scores_baseline, scores_future, output_dir, name_file, cluster)

        else:
            #global view only
            scores_baseline = confusion_matrix(presence_baseline)
            scores_future = confusion_matrix(presence_future)

            save_scores(scores_baseline, scores_future, output_dir, name_file)

    return


def read_in_data(data: str) -> np.ndarray:
    '''
    Function used to read in the data from R-file
    '''

    result = pyreadr.read_r(data)
    items = list(result.items())
    
    tmp = result[items[0][0]]
    tmp['cell_id'] = np.nan
    tmp.insert(0, 'index', tmp.index)
    res = tmp.to_numpy(dtype=float,copy=True)
    

    return res


def homogenize_data(phytoplankton_baseline: np.ndarray, zooplankton_baseline: np.ndarray, phytoplankton_projection: np.ndarray,
                    zooplankton_projection: np.ndarray, clusters: np.ndarray=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    This function homogenizes the given data
    '''

    #Check that data has the right format (minimal check)
    assert phytoplankton_baseline.shape[1] >= 5, 'Not enough entries in the phytoplankton baseline'
    assert zooplankton_baseline.shape[1] >= 5, 'Not enough entries in the zooplankton baseline'
    assert phytoplankton_projection.shape[1] >= 5, 'Not enough entries in the phytoplankton projection'
    assert zooplankton_projection.shape[1] >= 5, 'Not enough entries in the zooplankton projection'

    #select matrix with highest number of cells (might not be the best way to do this)
    if phytoplankton_projection.shape[0] > zooplankton_projection.shape[0]:
        id_matrix = phytoplankton_projection[:,[0,2,3]]  
    else:
        id_matrix = zooplankton_projection[:,[0,2,3]]

    #transform all datasets to the same format
    number_observations = max(phytoplankton_projection.shape[0],zooplankton_projection.shape[0])
    number_phytoplankton = phytoplankton_projection.shape[1]
    number_zooplankton = zooplankton_projection.shape[1]

    # allocate arrays for homogenized data
    clusters_extended = None
    if clusters is not None:
        clusters_extended = np.full((number_observations,4),np.nan)
    phyto_baseline_extended = np.full((number_observations,number_phytoplankton),np.nan)
    phyto_future_extended = np.full((number_observations,number_phytoplankton),np.nan)
    zoo_baseline_extended = np.full((number_observations,number_zooplankton),np.nan)
    zoo_future_extended = np.full((number_observations,number_zooplankton),np.nan)

    
    for i in tqdm(range(number_observations), desc='Copying data into homogenized array'):

        if i < phytoplankton_baseline.shape[0]:
            ind = np.argwhere((id_matrix[:,1] == phytoplankton_baseline[i,2]) & (id_matrix[:,2] == phytoplankton_baseline[i,3]))
            if ind.shape[0]>0:
                phyto_baseline_extended[ind,:] = phytoplankton_baseline[i,:]
        if i < phytoplankton_projection.shape[0]:
            ind = np.argwhere((id_matrix[:,1] == phytoplankton_projection[i,2]) & (id_matrix[:,2] == phytoplankton_projection[i,3]))
            if ind.shape[0]>0:
                phyto_future_extended[ind,:] = phytoplankton_projection[i,:]
        if i < zooplankton_baseline.shape[0]:
            ind = np.argwhere((id_matrix[:,1] == zooplankton_baseline[i,2]) & (id_matrix[:,2] == zooplankton_baseline[i,3]))
            if ind.shape[0]>0:
                zoo_baseline_extended[ind,:] = zooplankton_baseline[i,:]
        if i < zooplankton_projection.shape[0]:
            ind = np.argwhere((id_matrix[:,1] == zooplankton_projection[i,2]) & (id_matrix[:,2] == zooplankton_projection[i,3]))
            if ind.shape[0]>0:
                zoo_future_extended[ind,:] = zooplankton_projection[i,:]
        if clusters is not None:
            if i < clusters.shape[0]:
                ind = np.argwhere((id_matrix[:,1] == clusters[i,2]) & (id_matrix[:,2] == clusters[i,3]))
                if ind.shape[0]>0:
                    clusters_extended[ind,:] = clusters[i,[0,2,3,4]]
        
            
    #Clean data and only use values found in both baseline and future
    phyto_future_extended[np.isnan(phyto_baseline_extended[:,4]),4:] = np.nan
    phyto_baseline_extended[np.isnan(phyto_future_extended[:,4]),4:] = np.nan
    
    zoo_future_extended[np.isnan(zoo_baseline_extended[:,4]),4:]= np.nan
    zoo_baseline_extended[np.isnan(zoo_future_extended[:,4]),4:] = np.nan
    
    homogenized_baseline = np.hstack((phyto_baseline_extended,zoo_baseline_extended[:,4:]))
    homogenized_future = np.hstack((phyto_future_extended,zoo_future_extended[:,4:]))

    return homogenized_baseline, homogenized_future, clusters_extended

def get_presence_absence(homogenized_baseline: np.ndarray, homogenized_future: np.ndarray, threshold: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    This function converts a habitat suitability matrix into a presence/absence matrix based on a given threshold
    '''
        
    presence_baseline = (homogenized_baseline > threshold/100).astype(float)
    presence_baseline[np.isnan(homogenized_baseline)] = np.nan
    
    presence_future = (homogenized_future > threshold/100).astype(float)
    presence_future[np.isnan(homogenized_future)] = np.nan

    return presence_baseline, presence_future

def confusion_matrix(data: np.ndarray) -> np.ndarray:
    '''
    This function calculates the scores for each species pair.
    The data contains the presence of all species in a given region/cluster.
    The rows are the pixels available, and the columns are the different species
    '''

    _, cols = data.shape
    
    score = np.zeros((cols-4,cols-4))
    
    for j in tqdm(range(4,cols-1), desc='Calculating scores for species pairs'):

        sum_presence = np.sum(data[:,j]>0)

        score[j-4,j+1-4:] = [calculate_dunning(data[:,j]+data[:,k],data[:,j]-data[:,k],sum_presence) for k in range(j+1,cols)]

    return score

def save_scores(scores_baseline: np.ndarray, scores_future: np.ndarray, threshold: int, output_dir: str, name_file: str, cluster: np.ndarray=None) -> None:
    '''
    This function saves the scores given the threshold used to calculate the scores, and the clusters
    '''

    if cluster is not None:
        filename_baseline = f'{output_dir}Cluster/Cluster_{str(int(cluster))}_Threshold_{str(threshold)}_Scores_baseline_{name_file}'
        filename_future = f'{output_dir}Cluster/Cluster_{str(int(cluster))}_Threshold_{str(threshold)}_Scores_future_{name_file}'

        np.savetxt(filename_baseline, scores_baseline, delimiter=',')
        np.savetxt(filename_future, scores_future, delimiter=',')

    else:

        filename_baseline = f'{output_dir}Global/Threshold_{str(threshold)}_Scores_baseline_{name_file}'
        filename_future = f'{output_dir}Global/Threshold_{str(threshold)}_Scores_future_{name_file}'

        np.savetxt(filename_baseline, scores_baseline, delimiter=',')
        np.savetxt(filename_future, scores_future, delimiter=',')
    
    return

def calculate_dunning(presence_addition: np.ndarray, presence_subtraction: np.ndarray, presence_sum: int) -> np.ndarray: 
    '''
    This function calculates the associations and uses the findings in Evert et al., 2008
    to distinguish between positive and negative associations, and the absolute value expresses the significance
    '''

    co_occurrence = len(presence_addition[presence_addition == 2])
    absence_presence = len(presence_subtraction[presence_subtraction == -1])
    presence_absence = len(presence_subtraction[presence_subtraction == 1])
    co_absence = len(presence_addition[presence_addition == 0])
    

    check = np.array([[co_occurrence, absence_presence],
                       [presence_absence,co_absence]])

    number_tokens = np.sum(presence_addition >= 0)
    
    result = calculate_likelihood(check)

    expected = presence_sum*(co_occurrence + absence_presence)/number_tokens

    if(check[0,0] < expected):
        return -result
    return result
   

def calculate_shannon(vec: np.ndarray) -> float:
    '''
    This function calculates the shannon entropy from a vector.
    It is a measure for the uncertainty in the observed probability ditribution.
    '''
         
    n = np.sum(vec)
    shannon_entropy = np.sum(vec/n * np.log(vec/n + (vec == 0)))
              
    return shannon_entropy


def calculate_likelihood(contingency: np.ndarray) -> float:
    '''
    This function calculates the likelihood ratio given a contingency table.
    '''
    contingency_vectorized = np.reshape(contingency,(1, 4))

    likelihood = 2 * np.sum(contingency_vectorized) * (calculate_shannon(contingency_vectorized) -
                       calculate_shannon(np.sum(contingency,1)) - calculate_shannon(np.sum(contingency,0)))

    return  likelihood


        




    