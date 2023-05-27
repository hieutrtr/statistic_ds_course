from methods import cross_validation, scott, sturges, plotting
import os, json
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # read csv file
    df = pd.read_csv(os.path.abspath('abalone.csv'))
    # get the first 10 rows
    # df = df.head(10)
    # show columns
    print(df.columns)
    # choose fields for histogram
    fields = ['Length', 'Whole weight', 'Rings']
    # use scott method to choose bins
    scott_bins = scott.select_bin(df.to_dict(orient='records'), fields)
    print("Scott's method:")
    print(scott_bins)
    # use sturges method to choose bins
    sturges_bins = sturges.select_bin(df.to_dict(orient='records'), fields)
    print("Sturges's method:")
    print(sturges_bins)
    # use cross validation method to choose bins
    cross_validation_bins = cross_validation.select_bin(df, fields)
    print("Cross validation method:")
    print(cross_validation_bins)
    bins_dict = {
        "Sturges’s Rule": sturges_bins,
        "Scott’s normal reference rule": scott_bins,
        "Cross validation": cross_validation_bins
    }
    # plot multiple histograms
    plotting.plot_histogram(bins_dict, df.to_dict(orient='records'))
    # show plot
    plt.show()

