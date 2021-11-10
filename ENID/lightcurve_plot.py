import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def lightcurve_plot(data, source):
    # INPUT : 
    # data : light curve information, in the same dataframe format as obtained from alerce_retrieval
    # source : source name
    idx1 = np.where(data['fid'] == 1)[0]
    idx2 = np.where(data['fid'] == 2)[0]                                  
    # Plot the lightcurve for the input source in both G and R bands and display source name
    data_mjd = np.array(data['mjd'])
    data_lc = np.array(data['magpsf'])
    
    fig = plt.figure()
    plt.scatter(data_mjd[idx1], data_lc[idx1], c='green', label='G-band')
    plt.scatter(data_mjd[idx2], data_lc[idx2], c='red', label='R-band')
    plt.gca().invert_yaxis()
    plt.title(source, fontweight="bold")
    plt.xlabel('Time [MJD]')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid()
    plt.show()
