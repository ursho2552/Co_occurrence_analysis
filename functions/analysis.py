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


def find_baseline(file_names: list[str], models: list[str], algorithm_names: list[str], parameters: list[str], clusters: np.ndarray=None) -> np.ndarray:
    '''
    This function finds the baseline for each future projection based on the threshold, model, algorithm, and parameterset used.
    It also considers different clusters if they are provided
    '''    

    if clusters is None:
        clusters = [-1]

    list_files = np.zeros((len(file_names),3))
    list_files[:,0] = np.arange(len(file_names))
    thresholds = np.arange(25,41,1)
    thresholds_RF = np.arange(10,26,1)

    for threshold, threshold_RF in zip(thresholds, thresholds_RF):
        for model in models:
            for algorithm in algorithm_names:
                for parameter in parameters:
                    
                    if algorithm == 'RF':
                        threshold = threshold_RF
                    
                    future_name_suffix = f'Threshold_{str(threshold)}_Scores_future_{model}_{algorithm}_{parameter}.csv'
                    baseline_name_suffix = f'Threshold_{str(threshold)}_Scores_baseline_{model}_{algorithm}_{parameter}.csv'

                    for cluster in clusters:

                        if cluster > 0:
                            future_name = f'Cluster_{str(int(cluster))}_{future_name_suffix}'
                            baseline_name = f'Cluster_{str(int(cluster))}__{baseline_name_suffix}'
                        else:
                            future_name = future_name_suffix
                            baseline_name = baseline_name_suffix

                        ind_future = [ii for ii in range(len(file_names)) if future_name in file_names[ii]]
                        ind_baseline = [ii for ii in range(len(file_names)) if baseline_name in file_names[ii]]

                        list_files[ind_future,1] = 1
                        list_files[ind_future,2] = ind_baseline
        
    return list_files      



def get_difference(matrix_baseline: np.ndarray, matrix_projection: np.ndarray, threshold: int=75) -> list[int]:
    '''
    This function uses the likelihood ratio of species pairs in a baseline and a future projection
    to evaluate which pairs were lost, gained, remained constant, or where never there
    '''

    non_zero_LLR = matrix_baseline[matrix_baseline > 0]

    threshold_significance = np.percentile(non_zero_LLR, threshold)

    baseline = np.where(matrix_baseline > threshold_significance, 1, 0)
    future = np.where(matrix_projection > threshold_significance, 1, 0)
    
    
    loss = len(np.argwhere((baseline == 1) & (future == 0)))
    gain = len(np.argwhere((baseline == 0) & (future == 1)))
    const = len(np.argwhere((baseline == 1) & (future == 1)))
    never = len(np.argwhere((baseline == 0) & (future == 0)))
    
    return [loss,gain,const,never]

def flag_significant_pairs(matrix_baseline: np.ndarray, matrix_projection: np.ndarray, threshold: int=75) -> np.ndarray:
    '''
    This function flags species pairs gained in the projection compared to those in the baseline.
    The default threshold is the 75th percentile of the baseline values
    '''

    vec = matrix_baseline[matrix_baseline > 0]

    Y = np.percentile(vec, threshold)

    gained_pairs = np.where((matrix_baseline <= Y) & (matrix_projection > Y), 1, 0)
    
    return gained_pairs

    



