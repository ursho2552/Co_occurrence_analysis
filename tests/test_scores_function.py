#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 25 Sep 11:54:36 2023

@author: Urs Hofmann Elizondo
Unittest for functions in the scores module
"""

import unittest
import numpy as np

from tqdm import tqdm
from functools import partialmethod

from functions.scores import *

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
        expected_result = 4.498681156950466

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

    def test_homogenize_data_shape(self):
        
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

        result_baseline, result_future, _ = homogenize_data(phytoplankton_baseline, zooplankton_baseline,
                                                                                     phytoplankton_projection, zooplankton_projection, clusters)

        self.assertTrue(result_baseline.shape == result_future.shape)

        

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

        

if __name__ == '__main__':
    unittest.main()