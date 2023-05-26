import numpy as np

def sturges_rule(data):
    '''
    Function to calculate the number of bins to be used in a histogram using Sturges' Rule.
    data: Array of numerical data
    '''
    num_bins = 1 + np.log2(len(data))
    return num_bins

def select_bin(data, fields):
    '''
    Function to apply bin-selection-method to a specific field of a sample of dataset.
    data: Array of data with fields
    '''
    bins = {}
    for field in fields:
        # Select the field you want to apply bin-selection-method
        field_data = [record[field] for record in data]
        # Calculate the number of bins using Sturges' Rule
        num_bins = sturges_rule(field_data)
        bins[field] = num_bins
    return bins