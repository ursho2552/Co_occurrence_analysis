#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 25 Sep 11:54:36 2023

@author: Urs Hofmann Elizondo
Unittest for functions in the scores module
"""

import unittest
import numpy as np

import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
from functools import partialmethod

from functions.scores import *

import tempfile
import shutil

class MyTestCase(unittest.TestCase):

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    def test_calculate_shannon(self):

        test_vector = np.array([0,1])
        result = calculate_shannon(test_vector)

        self.assertEqual(result, 0.0)

    def test_calculate_shannon_empty(self):

        test_vector = np.array([0,0])
        result = calculate_shannon(test_vector)

        self.assertTrue(np.isnan(result))


    def test_calculate_likelihood(self):

        test_contingency = np.array([[1,0], [1,1]])
        result = calculate_likelihood(test_contingency)

        expected_result = 1.046496287529096
        self.assertAlmostEqual(result, expected_result, places = 4)

    def test_calculate_dunning(self):

        presences_species_1 = np.array([1, 0, 0, 0])
        presences_species_2 = np.array([1, 0, 0, 0])

        presence_addition = presences_species_1 + presences_species_2
        presence_subtraction = presences_species_1 - presences_species_2

        presence_sum = np.sum(presences_species_1)

        result = calculate_dunning(presence_addition, presence_subtraction, presence_sum)
        expected_result = 0.3669001403475045

        self.assertAlmostEqual(result, expected_result, places = 4)

    def test_confusion_matrix(self):

        np.random.seed(17)
        data = (np.random.rand(10,6)>0.5).astype(int)

        result = confusion_matrix(data).reshape((1,-1))
        expected_result = np.array([ [0., -1.26537409, 0.,  0.]])

        self.assertTrue(np.allclose(result, expected_result))

    def test_get_presence_absence(self):


        homogenized_baseline = np.array([0., 0.2, 0.4, 0.6, 0.8, 1.0])
        homogenized_future = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        threshold = 50

        result_baseline, result_future = get_presence_absence(homogenized_baseline, homogenized_future, threshold)

        expected_baseline = np.array([0., 0., 0., 1., 1., 1.])
        expected_future = np.array([0., 0., 0., 1., 1., 1.])

        self.assertTrue(np.allclose(result_baseline, expected_baseline))
        self.assertTrue(np.allclose(result_future, expected_future))


    def test_homogenize_data_baseline(self):

        phytoplankton_baseline = np.array([[0, 0, 0, 0, 0, 0]])

        zooplankton_baseline = np.array([[0, 0, 0, 0, 0, 0],
                                        [2, 2, 2, 2, 2, 2]])

        phytoplankton_projection = np.array([[0, 0, 0, 0, 0, 0],
                                            [1, 1, 1, 1, 1, 1],
                                            [2, 2, 2, 2, 2, 2],
                                            [3, 3, 3, 3, 3, 3]])

        zooplankton_projection = np.array([[0, 0, 0, 0, 0, 0],
                                            [2, 2, 2, 2, 2, 2],
                                            [3, 3, 3, 3, 3, 3]])

        clusters = None

        result_baseline, _, _ = homogenize_data(phytoplankton_baseline, zooplankton_baseline,
                                                                                     phytoplankton_projection, zooplankton_projection, clusters)

        result_baseline = result_baseline[np.isnan(result_baseline)==False]
        expected_baseline = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,   2.,  2.])

        self.assertTrue(np.allclose(result_baseline, expected_baseline))



    def test_homogenize_data_future(self):

        phytoplankton_baseline = np.array([[0, 0, 0, 0, 0, 0]])

        zooplankton_baseline = np.array([[0, 0, 0, 0, 0, 0],
                                        [2, 2, 2, 2, 2, 2]])

        phytoplankton_projection = np.array([[0, 0, 0, 0, 0, 0],
                                            [1, 1, 1, 1, 1, 1],
                                            [2, 2, 2, 2, 2, 2],
                                            [3, 3, 3, 3, 3, 3]])

        zooplankton_projection = np.array([[0, 0, 0, 0, 0, 0],
                                            [2, 2, 2, 2, 2, 2],
                                            [3, 3, 3, 3, 3, 3]])

        clusters = None

        _, result_future, clusters_extended = homogenize_data(phytoplankton_baseline, zooplankton_baseline,
                                                                                     phytoplankton_projection, zooplankton_projection, clusters)


        result_future = result_future[np.isnan(result_future)==False]
        expected_future = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 1.,  1.,  1.,  1., 2.,  2.,  2.,  2., 2.,  2., 3.,  3.,  3.,  3.])

        self.assertTrue(np.allclose(result_future, expected_future))


    def test_calculate_shannon_uniform_distribution(self):
        test_vector = np.array([1, 1, 1, 1])
        result = calculate_shannon(test_vector)
        expected_result = -1.3862943611  # ln(1/4) since it's a uniform distribution
        self.assertAlmostEqual(result, expected_result, places=4)

    def test_calculate_likelihood_zero_counts(self):
        test_contingency = np.array([[0, 0], [0, 0]])
        result = calculate_likelihood(test_contingency)
        self.assertTrue(np.isnan(result))  # Likelihood ratio is undefined

    def test_calculate_dunning_independent_species(self):
        presences_species_1 = np.array([1, 0, 0, 1])
        presences_species_2 = np.array([0, 1, 1, 0])

        presence_addition = presences_species_1 + presences_species_2
        presence_subtraction = presences_species_1 - presences_species_2
        presence_sum = np.sum(presences_species_1)

        result = calculate_dunning(presence_addition, presence_subtraction, presence_sum)
        self.assertLess(result, 0.0)  # No co-occurrence, so association should be 0


    def test_homogenize_data_empty_inputs(self):
        phytoplankton_baseline = np.empty((0, 6))
        zooplankton_baseline = np.empty((0, 6))
        phytoplankton_projection = np.empty((0, 6))
        zooplankton_projection = np.empty((0, 6))
        clusters = None

        result_baseline, result_future, result_clusters = homogenize_data(
            phytoplankton_baseline, zooplankton_baseline,
            phytoplankton_projection, zooplankton_projection, clusters
        )

        self.assertEqual(result_baseline.size, 0)
        self.assertEqual(result_future.size, 0)
        self.assertIsNone(result_clusters)

    def test_homogenize_data_mismatched_sizes(self):
        phytoplankton_baseline = np.array([[0, 0, 0, 0, 0, 0]])
        zooplankton_baseline = np.array([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])

        phytoplankton_projection = np.array([[0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2]])
        zooplankton_projection = np.array([[0, 0, 0, 0, 0, 0]])

        clusters = None

        result_baseline, result_future, _ = homogenize_data(
            phytoplankton_baseline, zooplankton_baseline,
            phytoplankton_projection, zooplankton_projection, clusters
        )

        self.assertTrue(result_baseline.shape[0] >= max(phytoplankton_projection.shape[0], zooplankton_projection.shape[0]))
        self.assertTrue(result_future.shape[0] >= max(phytoplankton_projection.shape[0], zooplankton_projection.shape[0]))


    def test_save_scores_invalid_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_dir = os.path.join(temp_dir, "nonexistent")
            scores_baseline = np.array([[1, 0], [0, 1]])
            scores_future = np.array([[0, 1], [1, 0]])

            with self.assertRaises(FileNotFoundError):
                save_scores(scores_baseline, scores_future, 50, invalid_dir, "test.csv")

if __name__ == '__main__':
    unittest.main()