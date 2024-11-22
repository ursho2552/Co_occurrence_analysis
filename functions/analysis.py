#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:10:29 2020

@author: ursho
"""
from typing import List, Optional
import numpy as np
import numpy.typing as npt

from functions.initialize import ConfigurationParameters


def find_baseline(
    file_names: List[str],
    models: List[str],
    algorithm_names: List[str],
    predictors: List[str],
    clusters: Optional[npt.NDArray[int]],
    configurations: ConfigurationParameters,
) -> npt.NDArray[int]:
    """
    Finds the baseline for each future projection based on the threshold,
    model, algorithm, and parameter set used. Also considers different clusters if provided.

    Args:
        file_names (List[str]): List of available file names.
        models (List[str]): List of models to consider.
        algorithm_names (List[str]): List of algorithm names.
        predictors (List[str]): List of predictors.
        clusters (Optional[npt.NDArray[int]]): Array of cluster identifiers.
        configurations (ConfigurationParameters): Configuration parameters with threshold ranges.

    Returns:
        npt.NDArray[int]: Array indicating matched indices for future and baseline files.
    """
    clusters = clusters if clusters is not None else np.array([-1], dtype=int)
    file_matrix = np.zeros((len(file_names), 3), dtype=int)  # Initialize as integer array
    file_matrix[:, 0] = np.arange(len(file_names))

    for model in models:
        for algo_idx, algorithm in enumerate(algorithm_names):
            # Extract threshold range for the current algorithm
            start_threshold = configurations.threshold_start[algo_idx]
            end_threshold = configurations.threshold_end[algo_idx]
            threshold_range = np.arange(start_threshold, end_threshold, 1)

            for threshold in threshold_range:
                for predictor in predictors:
                    # Construct suffixes for file names
                    future_suffix = f"Threshold_{threshold}_Scores_future_{model}_{algorithm}_{predictor}.csv"
                    baseline_suffix = f"Threshold_{threshold}_Scores_baseline_{model}_{algorithm}_{predictor}.csv"


                    for cluster in clusters:
                        # Adjust file names for cluster information
                        cluster_prefix = f"Cluster_{int(cluster)}_" if cluster > 0 else ""
                        future_name = f"{cluster_prefix}{future_suffix}"
                        baseline_name = f"{cluster_prefix}{baseline_suffix}"

                        # Identify indices of relevant files
                        future_indices = [
                            idx for idx, name in enumerate(file_names) if future_name in name
                        ]
                        baseline_indices = [
                            idx for idx, name in enumerate(file_names) if baseline_name in name
                        ]

                        # Update the file matrix with matches
                        for future_idx in future_indices:
                            file_matrix[future_idx, 1] = 1
                            file_matrix[future_idx, 2] = baseline_indices[0] if baseline_indices else -1

    return file_matrix

def get_difference(
    matrix_baseline: npt.NDArray[np.float64],
    matrix_projection: npt.NDArray[np.float64],
    threshold: int = 75,
) -> List[int]:
    """
    Evaluates changes in species pairs based on likelihood ratios.

    Args:
        matrix_baseline (np.ndarray): Baseline matrix of likelihood ratios.
        matrix_projection (np.ndarray): Future projection matrix of likelihood ratios.
        threshold (int): Percentile threshold for significance (default: 75).

    Returns:
        List[int]: Counts of lost, gained, constant, and never-present pairs.
    """
    # Calculate significance threshold
    if np.all(matrix_baseline == 0):
        significance_threshold = 0
    else:
        non_zero_values = matrix_baseline[matrix_baseline > 0]
        significance_threshold = np.percentile(non_zero_values, threshold)

    # Binary matrices for significant pairs
    baseline_binary = (matrix_baseline > significance_threshold).astype(int)
    future_binary = (matrix_projection > significance_threshold).astype(int)

    # Count changes
    loss = np.sum((baseline_binary == 1) & (future_binary == 0))
    gain = np.sum((baseline_binary == 0) & (future_binary == 1))
    const = np.sum((baseline_binary == 1) & (future_binary == 1))
    never = np.sum((baseline_binary == 0) & (future_binary == 0))

    return [loss, gain, const, never]


def flag_significant_pairs(
    matrix_baseline: npt.NDArray[np.float64],
    matrix_projection: npt.NDArray[np.float64],
    threshold: int = 75,
) -> npt.NDArray[int]:
    """
    Flags significant species pairs gained in the projection compared to the baseline.

    Args:
        matrix_baseline (np.ndarray): Baseline matrix of likelihood ratios.
        matrix_projection (np.ndarray): Future projection matrix of likelihood ratios.
        threshold (int): Percentile threshold for significance (default: 75).

    Returns:
        npt.NDArray[int]: Matrix with flags for newly significant pairs.
    """
    # Calculate significance threshold
    if np.all(matrix_baseline == 0):
        #all changes are gained pairs
        return (matrix_projection > 0).astype(int)

    significant_values = matrix_baseline[matrix_baseline > 0]
    significance_threshold = np.percentile(significant_values, threshold)

    # Flag gained pairs
    gained_pairs = np.where((matrix_baseline <= significance_threshold) & (matrix_projection > significance_threshold), 1, 0)

    return gained_pairs
