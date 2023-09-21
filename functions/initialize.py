#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thr Sept 21 15:39:48 2023

@author: ursho
"""

from dataclasses import dataclass

import yaml

@dataclass
class Configuration_parameters():
    directory_R_data: str
    R_data_file_root: str
    directory_clusters: str

    directory_analysis: str
    directory_output: str

    first_group: str
    second_group: str

    reference_time_name: str
    projection_time_name: str

    algorithm_names: list[str]
    parameter_names: list[str]

    threshold_start: list[int]
    threshold_end: list[int]

def read_config_file(configuration_file, configuration_class=Configuration_parameters):
    '''
    This function read the configuration file. It stores the values in configuration
    '''

    assert '.yaml' in configuration_file.lower(), "The configuration file should be a .yaml file"

    with open(configuration_file, 'r') as file:
        config_list = yaml.load(file, Loader=yaml.FullLoader)

    configuration = configuration_class(**config_list)

    return configuration


