#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:57:48 2020

@author: ursho
"""
from typing import List, Tuple, Optional
import logging
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import pyreadr

from functions.initialize import ConfigurationParameters


def calculate_scores(list_files: List[int], file_names: List[str], list_of_values: List[int],
                    clusters: np.ndarray, model_names: List[str], configurations: ConfigurationParameters) -> None:
    """
    Calculates scores for given thresholds and clusters.

    Args:
        list_files (np.ndarray): File index information.
        file_names (List[str]): List of file names.
        list_of_values (List[int]): Predictor, model, and algorithm indices.
        clusters (Optional[np.ndarray]): Cluster information.
        model_names (List[str]): Names of models.
        configurations (ConfigurationParameters): Configuration details.
    """

    algo_names = configurations.algorithm_names
    output_dir = configurations.directory_output
    predictor_value, model_value, algorithm_value = list_of_values

    # Locate indices of files
    file_indices = locate_file_indices(
        list_files, predictor_value, model_value, algorithm_value
    )
    phyto_baseline = read_in_data(file_names[file_indices["phyto_baseline"]])
    phyto_future = read_in_data(file_names[file_indices["phyto_future"]])
    zoo_baseline = read_in_data(file_names[file_indices["zoo_baseline"]])
    zoo_future = read_in_data(file_names[file_indices["zoo_future"]])

    name_file = f"{model_names[model_value - 1]}_{algo_names[algorithm_value - 1]}_p{predictor_value}.csv"
    logging.info(name_file)

    homogenized_baseline, homogenized_future, homogenized_clusters = homogenize_data(phyto_baseline,
                                                                                      zoo_baseline,
                                                                                      phyto_future,
                                                                                      zoo_future,
                                                                                      clusters=clusters)

    # Process thresholds
    process_thresholds(
        homogenized_baseline,
        homogenized_future,
        homogenized_clusters,
        configurations,
        algorithm_value,
        output_dir,
        name_file,
    )


def locate_file_indices(
    list_files: np.ndarray, predictor_value: int, model_value: int, algorithm_value: int
) -> dict:
    """
    Finds file indices for phyto and zoo data.

    Args:
        list_files (np.ndarray): File index array.
        predictor_value (int): Predictor index.
        model_value (int): Model index.
        algorithm_value (int): Algorithm index.

    Returns:
        dict: Indices for phyto baseline, phyto future, zoo baseline, zoo future.
    """
    return {
        "phyto_baseline": np.where(
            (list_files[:, 1] == 1)
            & (list_files[:, 2] == 1)
            & (list_files[:, 3] == algorithm_value)
            & (list_files[:, 4] == 0)
            & (list_files[:, 5] == predictor_value)
        )[0][0],
        "phyto_future": np.where(
            (list_files[:, 1] == 1)
            & (list_files[:, 2] == 0)
            & (list_files[:, 3] == algorithm_value)
            & (list_files[:, 4] == model_value)
            & (list_files[:, 5] == predictor_value)
        )[0][0],
        "zoo_baseline": np.where(
            (list_files[:, 1] == 0)
            & (list_files[:, 2] == 1)
            & (list_files[:, 3] == algorithm_value)
            & (list_files[:, 4] == 0)
            & (list_files[:, 5] == predictor_value)
        )[0][0],
        "zoo_future": np.where(
            (list_files[:, 1] == 0)
            & (list_files[:, 2] == 0)
            & (list_files[:, 3] == algorithm_value)
            & (list_files[:, 4] == model_value)
            & (list_files[:, 5] == predictor_value)
        )[0][0],
    }

def process_thresholds(
    homogenized_baseline: np.ndarray,
    homogenized_future: np.ndarray,
    homogenized_clusters: Optional[np.ndarray],
    configurations: ConfigurationParameters,
    algorithm_value: int,
    output_dir: str,
    name_file: str,
) -> None:
    """
    Processes thresholds for baseline and future data.

    Args:
        homogenized_baseline (np.ndarray): Baseline data.
        homogenized_future (np.ndarray): Future data.
        homogenized_clusters (Optional[np.ndarray]): Cluster data.
        configurations (ConfigurationParameters): Configuration details.
        algorithm_value (int): Algorithm index.
        output_dir (str): Output directory.
        name_file (str): Name of the output file.
    """
    start_threshold = configurations.threshold_start[algorithm_value - 1]
    end_threshold = configurations.threshold_end[algorithm_value - 1]
    threshold_range = np.arange(start_threshold, end_threshold, 1)

    for thr in tqdm(threshold_range, desc="Processing thresholds"):
        presence_baseline, presence_future = get_presence_absence(
            homogenized_baseline, homogenized_future, thr
        )

        if homogenized_clusters is not None:
            process_clusters(
                presence_baseline,
                presence_future,
                homogenized_clusters,
                thr,
                output_dir,
                name_file,
            )
        else:
            scores_baseline = confusion_matrix(presence_baseline)
            scores_future = confusion_matrix(presence_future)
            save_scores(scores_baseline, scores_future, thr, output_dir, name_file)

def process_clusters(
    presence_baseline: np.ndarray,
    presence_future: np.ndarray,
    clusters: np.ndarray,
    threshold: int,
    output_dir: str,
    name_file: str,
) -> None:
    """
    Processes clusters to calculate and save scores.

    Args:
        presence_baseline (np.ndarray): Baseline presence/absence matrix.
        presence_future (np.ndarray): Future presence/absence matrix.
        clusters (np.ndarray): Cluster data.
        threshold (int): Threshold for calculations.
        output_dir (str): Output directory.
        name_file (str): Name of the output file.
    """
    cluster_ids = np.unique(clusters[~np.isnan(clusters[:, -1]), -1])

    for cluster_id in cluster_ids:
        cluster_baseline = presence_baseline[clusters[:, -1] == cluster_id]
        cluster_future = presence_future[clusters[:, -1] == cluster_id]

        scores_baseline = confusion_matrix(cluster_baseline)
        scores_future = confusion_matrix(cluster_future)

        save_scores(
            scores_baseline, scores_future, threshold, output_dir, name_file, cluster_id
        )

def read_in_data(file_path: str) -> np.ndarray:
    """
    Reads data from an R file using pyreadr.

    Args:
        file_path (str): Path to the R file.

    Returns:
        np.ndarray: Data as a NumPy array.
    """
    result = pyreadr.read_r(file_path)
    if not result:
        raise ValueError(f"No data found in file: {file_path}")

    # Extract data
    key, data_frame = next(iter(result.items()))
    data_frame["cell_id"] = np.nan
    data_frame.insert(0, "index", data_frame.index)
    return data_frame.to_numpy(dtype=float, copy=True)

def homogenize_data(phytoplankton_baseline: np.ndarray, zooplankton_baseline: np.ndarray,
                    phytoplankton_projection: np.ndarray, zooplankton_projection: np.ndarray,
                    clusters: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    This function homogenizes the given data
    '''

    #Check that data has the right format (minimal check)
    assert phytoplankton_baseline.shape[1] > 5, 'Not enough entries in the phytoplankton baseline'
    assert zooplankton_baseline.shape[1] > 5, 'Not enough entries in the zooplankton baseline'
    assert phytoplankton_projection.shape[1] > 5, 'Not enough entries in the phytoplankton projection'
    assert zooplankton_projection.shape[1] > 5, 'Not enough entries in the zooplankton projection'

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

def get_presence_absence(homogenized_baseline: np.ndarray,
                        homogenized_future: np.ndarray,
                        threshold: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    This function converts a habitat suitability matrix into a
    presence/absence matrix based on a given threshold
    '''

    assert threshold >= 0, 'Threshold has to be non-negative.'
    assert threshold <= 100, 'Threshold has to be between 1 and 100.'

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

    rows, cols = data.shape

    assert cols >= 6, 'Incorrect data format. The first columns should be ID|Latitude|Longitude|Month, and should contain at least two species'
    assert rows > 1, 'Not enough observations in the dataset'

    score = np.zeros((cols-4,cols-4))

    for j in tqdm(range(4,cols-1), desc='Calculating scores for species pairs'):

        sum_presence = np.sum(data[:,j]>0)

        score[j-4,j+1-4:] = [calculate_dunning(data[:,j]+data[:,k],data[:,j]-data[:,k],sum_presence) for k in range(j+1,cols)]

    return score

def save_scores(scores_baseline: np.ndarray, scores_future: np.ndarray, threshold: int,
                output_dir: str, name_file: str, cluster: int=None) -> None:
    '''
    This function saves the scores given the threshold used to calculate the scores,
    and the clusters
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


def calculate_dunning(presence_addition: npt.NDArray[int], presence_subtraction: npt.NDArray[int],
                      presence_sum: int) -> np.float64:
    '''
    This function calculates the associations and uses the findings in Evert et al., 2008
    to distinguish between positive and negative associations, and the absolute value
    expresses the significance
    '''

    # Check if species patterns are identical
    # if np.all(presence_addition == 2) and np.all(presence_subtraction == 0):
    #     return np.inf

    co_occurrence = len(presence_addition[presence_addition == 2])
    absence_presence = len(presence_subtraction[presence_subtraction == -1])
    presence_absence = len(presence_subtraction[presence_subtraction == 1])
    co_absence = len(presence_addition[presence_addition == 0])

    #insert at least one to get more accurate results that take into account complete overlap
    number_tokens = len(presence_addition)

    if absence_presence == 0:
        absence_presence = 1
        number_tokens += 1

    if presence_absence == 0:
        presence_absence = 1
        number_tokens += 1

    check = np.array([[co_occurrence, absence_presence],
                       [presence_absence,co_absence]])

    result = calculate_likelihood(check)

    expected = presence_sum*(co_occurrence + absence_presence)/number_tokens

    if(check[0,0] < expected):
        return -result
    return result


def calculate_shannon(vec: npt.NDArray[int]) -> float:
    '''
    This function calculates the shannon entropy from a vector.
    It is a measure for the uncertainty in the observed probability ditribution.
    '''

    total = np.sum(vec)
    if total > 0:
        shannon_entropy = np.sum(vec/total * np.log(vec/total + (vec == 0)))
    else:
        shannon_entropy = np.nan

    return shannon_entropy


def calculate_likelihood(contingency: npt.NDArray[int]) -> float:
    '''
    This function calculates the likelihood ratio given a contingency table.
    '''
    contingency_vectorized = np.reshape(contingency,(1, 4))

    likelihood = 2 * np.sum(contingency_vectorized) * (calculate_shannon(contingency_vectorized) -
                calculate_shannon(np.sum(contingency, 1)) -
                calculate_shannon(np.sum(contingency, 0)))

    return  likelihood
