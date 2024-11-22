#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thr Sept 21 15:39:48 2023

@author: ursho
"""

from dataclasses import dataclass
from typing import List, Type, TypeVar
import yaml

# Type variable for generality in the configuration class
T = TypeVar('T', bound='ConfigurationParameters')

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

    algorithm_names: List[str]
    predictor_set: List[str]

    threshold_start: List[int]
    threshold_end: List[int]

def read_config_file(
    configuration_file: str,
    configuration_class: Type[T] = ConfigurationParameters
) -> T:
    """
    Reads the configuration file and stores its values in the specified configuration class.

    Args:
        configuration_file (str): Path to the configuration file (must be a .yaml file).
        configuration_class (Type[T]): The class to store the configuration (default: ConfigurationParameters).

    Returns:
        T: An instance of the configuration class populated with data from the YAML file.

    Raises:
        AssertionError: If the configuration file is not a .yaml file or if the thresholds mismatch.
    """
    # Ensure the file has a .yaml extension
    assert configuration_file.lower().endswith('.yaml'), "The configuration file should be a .yaml file"

    # Read and parse the YAML file
    with open(configuration_file, 'r', encoding='utf-8') as file:
        config_data = yaml.load(file, Loader=yaml.FullLoader)

    # Instantiate the configuration class
    configuration = configuration_class(**config_data)

    # Ensure threshold lists have matching lengths
    assert len(configuration.threshold_start) == len(configuration.threshold_end), (
        "The number of starting points and end points does not match."
    )

    return configuration
