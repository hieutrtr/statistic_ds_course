import numpy as np

def Scott_bins(data):
    n = len(data)
    sigma = np.std(data)
    bin_width = 3.49 * sigma / np.power(n, 1/3)
    k = np.ceil((np.max(data) - np.min(data)) / bin_width) + 1
    return int(k)

def select_bin(data, fields):
    
    #Create a dictionary to store the bin width and number of bins for each field
    bins = {}
    
    for field in fields:
        #Get the values of the field
        values = [item[field] for item in data]
        #Calculate the bin width and number of bins
        num_bins = Scott_bins(values)
        
        #Store the bin width and number of bins
        bins[field] = num_bins
        
    #Return the dictionary
    return bins