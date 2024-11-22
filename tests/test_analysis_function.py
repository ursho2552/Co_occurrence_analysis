#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unittests for functions in the analysis module.
"""

import unittest
import numpy as np

import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.analysis import find_baseline, get_difference, flag_significant_pairs
from functions.initialize import ConfigurationParameters

class TestAnalysisFunctions(unittest.TestCase):

    def test_find_baseline_no_clusters(self):
        file_names = [
            "Threshold_10_Scores_future_model1_algo1_pred1.csv",
            "Threshold_10_Scores_baseline_model1_algo1_pred1.csv"
        ]
        models = ["model1"]
        algorithms = ["algo1"]
        predictors = ["pred1"]
        clusters = None
        configurations = ConfigurationParameters(
            directory_r_data="",
            r_data_file_root="",
            directory_clusters="",
            directory_analysis="",
            directory_output="",
            first_group="",
            second_group="",
            reference_time_name="",
            projection_time_name="",
            algorithm_names=algorithms,
            predictor_set=predictors,
            threshold_start=[10],
            threshold_end=[11]
        )

        result = find_baseline(file_names, models, algorithms, predictors, clusters, configurations)
        expected = np.array([[0, 1, 1], [1, 0, 0]])  # File matrix with matches

        self.assertTrue(np.array_equal(result, expected))

    def test_find_baseline_with_clusters(self):
        file_names = [
            "Cluster_1_Threshold_10_Scores_future_model1_algo1_pred1.csv",
            "Cluster_1_Threshold_10_Scores_baseline_model1_algo1_pred1.csv",
        ]
        models = ["model1"]
        algorithms = ["algo1"]
        predictors = ["pred1"]
        clusters = np.array([1])
        configurations = ConfigurationParameters(
            directory_r_data="",
            r_data_file_root="",
            directory_clusters="",
            directory_analysis="",
            directory_output="",
            first_group="",
            second_group="",
            reference_time_name="",
            projection_time_name="",
            algorithm_names=algorithms,
            predictor_set=predictors,
            threshold_start=[10],
            threshold_end=[11]
        )

        result = find_baseline(file_names, models, algorithms, predictors, clusters, configurations)
        expected = np.array([[0, 1, 1], [1, 0, 0]])  # File matrix with matches

        self.assertTrue(np.array_equal(result, expected))

    def test_get_difference(self):
        baseline = np.array([[0.8, 0.1], [0.4, 0.9]])
        projection = np.array([[0.9, 0.2], [0.3, 0.8]])
        threshold = 75  # 75th percentile threshold

        result = get_difference(baseline, projection, threshold)
        expected = [1, 1, 0, 2]  # loss, gain, constant, never

        self.assertEqual(result, expected)

    def test_get_difference_empty_matrices(self):
        baseline = np.zeros((2, 2))
        projection = np.zeros((2, 2))

        result = get_difference(baseline, projection)
        expected = [0, 0, 0, 4]  # All never-present

        self.assertEqual(result, expected)

    def test_flag_significant_pairs(self):
        baseline = np.array([[0.1, 0.2], [0.0, 0.7]])
        projection = np.array([[0.2, 0.9], [0.3, 0.8]])
        threshold = 75  # 75th percentile threshold

        result = flag_significant_pairs(baseline, projection, threshold)
        expected = np.array([[0, 1], [0, 0]])  # Newly significant pairs

        self.assertTrue(np.array_equal(result, expected))

    def test_flag_significant_pairs_no_gain(self):
        baseline = np.array([[0.8, 0.9], [0.7, 1.0]])
        projection = np.array([[0.8, 0.9], [0.7, 1.0]])

        result = flag_significant_pairs(baseline, projection)
        expected = np.zeros_like(baseline)  # No newly significant pairs

        self.assertTrue(np.array_equal(result, expected))

    def test_flag_significant_pairs_empty(self):
        baseline = np.zeros((2, 2))
        projection = np.zeros((2, 2))

        result = flag_significant_pairs(baseline, projection)
        expected = np.zeros_like(baseline)  # No newly significant pairs

        self.assertTrue(np.array_equal(result, expected))


if __name__ == "__main__":
    unittest.main()
