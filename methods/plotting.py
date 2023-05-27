import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(bins_dict, data):
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 4))
    # extract data from dataset
    field_data = {}
    for field in bins_dict['Sturges’s Rule'].keys():
        field_data[field] = [row[field] for row in data]
    i = 0
    for method, bins in bins_dict.items():
        j = 0
        for field, bin in bins.items():
            # Plotting histograms
            axes[i][j].hist(field_data[field], bins=bin, alpha=0.5, label=field)
            axes[i][j].set_title('Method: {} | field: {} | bin {}'.format(method, field, bin))
            axes[i][j].legend()
            j += 1
        i += 1
    # Adjust spacing between subplots
    plt.tight_layout()
    # Show the plot
    plt.show()
