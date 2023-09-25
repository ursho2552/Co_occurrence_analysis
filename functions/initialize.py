#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thr Sept 21 15:39:48 2023

@author: ursho
"""

from dataclasses import dataclass

import yaml

@dataclass
class ConfigurationParameters():
    '''
    Dataclass defining the parameters used in the analysis
    '''
    directory_r_data: str
    r_data_file_root: str
    directory_clusters: str

    directory_analysis: str
    directory_output: str

    first_group: str
    second_group: str

    reference_time_name: str
    projection_time_name: str

    algorithm_names: list[str]
    predictor_set: list[str]

    threshold_start: list[int]
    threshold_end: list[int]

def read_config_file(configuration_file, configuration_class=ConfigurationParameters):
    '''
    This function read the configuration file. It stores the values in configuration
    '''

    assert '.yaml' in configuration_file.lower(), "The configuration file should be a .yaml file"

    with open(configuration_file, 'r', encoding='utf-8') as file:
        config_list = yaml.load(file, Loader=yaml.FullLoader)

    configuration = configuration_class(**config_list)

    # check that the entries match
    assert len(configuration.threshold_start) == len(configuration.threshold_end), 'The number of starting points and end points does not match.'

    return configuration
