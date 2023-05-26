import numpy as np

def Scott_bins(data):
    # Calculate the interquartile range.
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25

    # Calculate the optimal bin width.
    bin_width = 2 * (iqr/len(data)**(1/3))

    # Calculate the number of bins.
    num_bins = int((max(data) - min(data))/bin_width)

    # Return the optimal bin width and number of bins.
    return bin_width, num_bins

def select_bin(data, fields):
    
    #Create a dictionary to store the bin width and number of bins for each field
    bins = {}
    
    for field in fields:
        #Get the values of the field
        values = [item[field] for item in data]
        print(values)
        #Calculate the bin width and number of bins
        _, num_bins = Scott_bins(values)
        
        #Store the bin width and number of bins
        bins[field] = num_bins
        
    #Return the dictionary
    return bins