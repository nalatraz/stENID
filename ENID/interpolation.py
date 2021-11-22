from george import kernels
import george
from scipy.optimize import minimize
import pickle
import torch
import torch.nn.functional as F
import numpy as np





def GP_simple(x, y, yerr, plot=False):
    
    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(y)

    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y)
    
    kernel = np.var(y) * kernels.ExpSquaredKernel(20)
    gp = george.GP(kernel)
    gp.compute(x, yerr)
    
    x_pred = np.arange(min(x), max(x), 0.25)
    #x_pred = np.linspace(min(x), max(x), 1000)
    
    pred, pred_var = gp.predict(y, x_pred, return_var=True)
    
    result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
    p0 = gp.get_parameter_vector()

    gp.set_parameter_vector(result.x)

    pred, pred_var = gp.predict(y, x_pred, return_var=True)
    pred_error = np.sqrt(pred_var)
    
    if plot == True:
        plt.figure()
        plt.fill_between(x_pred, pred - pred_error, pred + pred_error,
                        color="k", alpha=0.2)
        plt.plot(x_pred, pred, "k", lw=1.5, alpha=0.5)
        plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
        plt.xlabel("Days")
        plt.ylabel("Normalised Flux")
        #plt.gca().invert_yaxis()
        plt.show()
    
    return x_pred, pred, pred_error
    
def mag2flux_normalised(mag, mag_error):
    
    # Generate arrays for storage
    lc_flux = np.array([])
    lc_error = np.array([])
    
    # Convert magnitudes to flux. We have assumed zero points equal to 0.
    for i in range(len(mag)):
        flux = 10 ** (- mag[i] / 2.5)
        flux_error = flux * mag_error[i] / 1.0857
        
        lc_flux = np.append(lc_flux, flux)
        lc_error = np.append(lc_error, flux_error)
            
    # Normalise the flux so that the peak value is 1
    scale_factor = np.nanmax(lc_flux)
    
    lc_flux = lc_flux / scale_factor
    lc_error = lc_error / scale_factor
    
    return lc_flux, lc_error, scale_factor

def flux2mag(flux, flux_error):
    
    # Generate arrays for storage
    lc_mag = np.array([])
    lc_mag_error = np.array([])
    
    # Convert flux to magnitude. We have assumed zero points equal to 0.
    for i in range(len(flux)):
        
        lc_mag = np.append(lc_mag, -2.5*np.log10(flux[i]))
        lc_mag_error = np.append(lc_mag_error, 1.0857 * flux_error[i] / flux[i])
        
    return lc_mag, lc_mag_error

def interpolate(magnitude, magnitude_error, mjd):
    
    # Compute the normalised flux
    flux, flux_error, scale = mag2flux_normalised(magnitude, magnitude_error)
    
    # Interpolate the lightcurve with a simple Gaussian Process
    xpred, pred, pred_error = GP_simple(mjd, flux, flux_error)
    
    pred *= scale
    pred_error *= scale 
    
    # Convert back to magnitudes
    pred, pred_error = flux2mag(pred, pred_error)
    
    return pred, pred_error, xpred

def label_encoding(labels):
    print(labels)
    # Ignore the TDEs for the moment. 
    # They have label -1, but this can be changed if we want to study these.
    idx = np.where(labels[:,1] != -1)
    
    # Determine the number of classes
    num_classes = len(np.unique(labels[idx]))
    
    # Encode the array in a tensor
    labels_torch = torch.tensor(labels[idx])
    labels_encoded = F.one_hot(labels_torch, num_classes=num_classes)
    
    return labels_encoded

def label_translate(labels):
    # Load the class translator
    dictionary_filename = 'ENID/class_dictionary.pickle'
    file = open(dictionary_filename, "rb")
    dictionary = pickle.load(file)
    
    # Get names of accepted classes
    keys_list = dictionary.keys()
    
    # Translate classes
    labels_translated = [dictionary[label[0]] for label in labels if label[0] in keys_list]
    
    # Format in a numpy array for future processing
    labels_translated = np.array(labels_translated)
    
    return labels_translated

def preprocessing(filename):
    
    # Load the data
    file = open(filename, "rb")
    datadict = pickle.load(file)
    
    m = len(datadict['Data'])
    l = 6000
    
    # Initialise custom python arrays
    data_initialised = np.zeros((2*m,l)) + np.nan
    error_initialised = np.zeros((2*m,l)) + np.nan
    mjd_initialised = np.zeros((2*m,l)) + np.nan
    
    # Steps in this function : 
    # 1. Combine detections and non detections
    datadict = dettwopredet(datadict, 14, 2)
    fails = []
    
    i = 0
    j = 0
    broken = False
    
    while i < m:
        
        # 2. Interpolate the data
        try:
            lightcurve_R, error_R, mjd_R = interpolate(datadict['Data'][i]['R_mag_wn'], datadict['Data'][i]['R_err_wn'], datadict['Data'][i]['R_mjd_wn'])
            lightcurve_G, error_G, mjd_G = interpolate(datadict['Data'][i]['G_mag_wn'], datadict['Data'][i]['G_err_wn'], datadict['Data'][i]['G_mjd_wn'])
            
            # 3. Group lightcurves in a global array
            n_R = len(lightcurve_R)
            n_G = len(lightcurve_G)
        
            if n_R > l-1:
                print('\nERROR : Lightcurve dimensions exceed the allowed size of ' + str(l) + '. Intialise the global array with a size larger than ' + str(n_R) + '.\n')
                broken = True
                break
        
            elif n_G > l-1:
                print('\nERROR : Lightcurve dimensions exceed the allowed size of ' + str(l) + '. Intialise the global array with a size larger than ' + str(n_G) + '.\n')
                broken = True
                break
            
            else:
                data_initialised[j, 0:n_R] = lightcurve_R
                error_initialised[j, 0:n_R] = error_R
                mjd_initialised[j, 0:n_R] = mjd_R
        
                data_initialised[j+1, 0:n_G] = lightcurve_G
                error_initialised[j+1, 0:n_G] = error_G
                mjd_initialised[j+1, 0:n_G] = mjd_G
                    
        except:
            print('\nERROR : Interpolation failed. Discarding entry.\n')
            fails.append([i, datadict['Name'][i], datadict['Label'][i]])
            
        i += 1
        j += 2
    
    if broken == True:
        data = -1
        labels = -1
        
    else:
        # 4. Trim array down to longest lightcurve
        count = [np.count_nonzero(np.isnan(data_initialised[i,:])) for i in range(m)]
        col = l - max(count)
        data = data_initialised[:, 0:col-1]
        error = error_initialised[:, 0:col-1]
        
        # 5. Structure labels
        labeldict = datadict['Label']
        labels = label_translate(labeldict)

    print('Saving Arrays...')
    np.save('data_lc_1', data)
    np.save('labels_1', labels)
    np.save('data_err_1', error)
    np.save('fails_1', fails)
                    
    return data, labels, error, fails

def label_count(labels_list):
    labels_uncoded = array()

    for i in range(len(labels_list)):
        labels_uncoded.update([labels_list[i][0]])
    
    count = Counter(labels_uncoded.data)
    
    return count

def dettwopredet(lightcurves,timediff,noofpoint):
    # Listing the lists, so that they work later
    R_non_mjd2_bool = []
    G_non_mjd2_bool = []

    R_non_mjd_additional = []
    R_non_mag_additional = []
    G_non_mjd_additional = []
    G_non_mag_additional = []

    # Loop start to work with all of them
    for i in range(len(lightcurves['Data'])): # Go through each light curve
    #for i in range(400):
        # Save old non-detections
        lightcurves['Data'][i]['R_non_org'] = lightcurves['Data'][i]['R_non']
        lightcurves['Data'][i]['G_non_org'] = lightcurves['Data'][i]['G_non']

        ## R: Red Dead Redemption Edition

        # Remove all non-detections that have occured at the same time as a real detection
        if type(lightcurves['Data'][i]['R_non']['mjd']) == np.ndarray: # Check if it is an array = has multipl evalue
            R_non_set = list(set(lightcurves['Data'][i]['R_mjd']).intersection(lightcurves['Data'][i]['R_non']['mjd'])) # Then find the intersection between non-detections and detections

            if len(R_non_set) > 0: # if it is non-empty
                lightcurves['Data'][i]['R_non']['mag'] = lightcurves['Data'][i]['R_non']['mag'][~(lightcurves['Data'][i]['R_non']['mjd']==R_non_set)]
                lightcurves['Data'][i]['R_non']['mjd'] = lightcurves['Data'][i]['R_non']['mjd'][~(lightcurves['Data'][i]['R_non']['mjd']==R_non_set)]

        else:
            R_non_set = list(set(lightcurves['Data'][i]['R_mjd']).intersection([lightcurves['Data'][i]['R_non']['mjd']]))

            if len(R_non_set) > 0:
                lightcurves['Data'][i]['R_non']['mag'] = lightcurves['Data'][i]['R_non']['mag'][~(lightcurves['Data'][i]['R_non']['mjd']==R_non_set)]
                lightcurves['Data'][i]['R_non']['mjd'] = lightcurves['Data'][i]['R_non']['mjd'][~(lightcurves['Data'][i]['R_non']['mjd']==R_non_set)]

        # Check if code has put in a -1:

        if type(lightcurves['Data'][i]['R_non']['mjd']) == int:
            # Give all of them an empty array so that it works
            R_non_mjd2_bool.append([float('nan')])
            R_non_mjd_additional.append([float('nan')])
            R_non_mag_additional.append([float('nan')])

            # Put in the actual lightcurves
            lightcurves['Data'][i]['R_mag_wn'] = lightcurves['Data'][i]['R_mag']
            lightcurves['Data'][i]['R_mjd_wn'] = lightcurves['Data'][i]['R_mjd']
            lightcurves['Data'][i]['R_err_wn'] = lightcurves['Data'][i]['R_err']

        else:
            # Start date
            R_start = lightcurves['Data'][i]['R_mjd'][0] # Find the start date

            # Find the non-detections that lies timediff before we start
            R_non_mjd2_bool.append((lightcurves['Data'][i]['R_non']['mjd'] > R_start - timediff) & (lightcurves['Data'][i]['R_non']['mjd'] < R_start)) # For those where we don't need to check for magdiff        

            R_non_mjd_additional.append(lightcurves['Data'][i]['R_non']['mjd'][R_non_mjd2_bool[i]])
            R_non_mag_additional.append(lightcurves['Data'][i]['R_non']['mag'][R_non_mjd2_bool[i]])

            # Get only the last two non-detections - as long as they have come within timediff

            R_non_mjd_additional[i] = R_non_mjd_additional[i][-noofpoint:]
            R_non_mag_additional[i] = R_non_mag_additional[i][-noofpoint:]

            # The new data vectors are added to the existing vectors

            lightcurves['Data'][i]['R_mag_wn'] = np.append(R_non_mag_additional[i],lightcurves['Data'][i]['R_mag'])
            lightcurves['Data'][i]['R_mjd_wn'] = np.append(R_non_mjd_additional[i],lightcurves['Data'][i]['R_mjd'])

            # Remove NaNs
            lightcurves['Data'][i]['R_mag_wn'] = lightcurves['Data'][i]['R_mag_wn'][~np.isnan(lightcurves['Data'][i]['R_mag_wn'])] # Removes NaNs
            lightcurves['Data'][i]['R_mjd_wn'] = lightcurves['Data'][i]['R_mjd_wn'][~np.isnan(lightcurves['Data'][i]['R_mjd_wn'])]

            # A new error list is added as well, with magerror being put in
            lightcurves['Data'][i]['R_err_wn'] = np.append(np.repeat(-1,len(R_non_mag_additional[i])),lightcurves['Data'][i]['R_err'])
            lightcurves['Data'][i]['R_err_wn'] = lightcurves['Data'][i]['R_err_wn'][~np.isnan(lightcurves['Data'][i]['R_err_wn'])] # remove NaN again

        ## G: Green New Deal Edition
        # Remove all non-detections that have occured at the same time as a real detection
        if type(lightcurves['Data'][i]['G_non']['mjd']) == np.ndarray: # Check if it is an array = has multipl evalue
            G_non_set = list(set(lightcurves['Data'][i]['G_mjd']).intersection(lightcurves['Data'][i]['G_non']['mjd'])) # Then find the intersection between non-detections and detections

            if len(G_non_set) > 0: # if it is non-empty
                lightcurves['Data'][i]['G_non']['mag'] = lightcurves['Data'][i]['G_non']['mag'][~(lightcurves['Data'][i]['G_non']['mjd']==G_non_set)]
                lightcurves['Data'][i]['G_non']['mjd'] = lightcurves['Data'][i]['G_non']['mjd'][~(lightcurves['Data'][i]['G_non']['mjd']==G_non_set)]

        else:
            G_non_set = list(set(lightcurves['Data'][i]['G_mjd']).intersection([lightcurves['Data'][i]['G_non']['mjd']]))

            if len(G_non_set) > 0:
                lightcurves['Data'][i]['G_non']['mag'] = lightcurves['Data'][i]['G_non']['mag'][~(lightcurves['Data'][i]['G_non']['mjd']==G_non_set)]
                lightcurves['Data'][i]['G_non']['mjd'] = lightcurves['Data'][i]['G_non']['mjd'][~(lightcurves['Data'][i]['G_non']['mjd']==G_non_set)]

        # Check if code has put in a -1:

        if type(lightcurves['Data'][i]['G_non']['mjd']) == int:
            # Give all of them an empty array so that it works
            G_non_mjd2_bool.append([float('nan')])
            G_non_mjd_additional.append([float('nan')])
            G_non_mag_additional.append([float('nan')])

            # Put in the actual lightcurves
            lightcurves['Data'][i]['G_mag_wn'] = lightcurves['Data'][i]['G_mag']
            lightcurves['Data'][i]['G_mjd_wn'] = lightcurves['Data'][i]['G_mjd']
            lightcurves['Data'][i]['G_err_wn'] = lightcurves['Data'][i]['G_err']

        else:
            # Start date
            G_start = lightcurves['Data'][i]['G_mjd'][0] # Find the start date

            # Find the non-detections that lies timediff before we start
            G_non_mjd2_bool.append((lightcurves['Data'][i]['G_non']['mjd'] > G_start - timediff) & (lightcurves['Data'][i]['G_non']['mjd'] < G_start)) # For those where we don't need to check for magdiff

            G_non_mjd_additional.append(lightcurves['Data'][i]['G_non']['mjd'][G_non_mjd2_bool[i]])
            G_non_mag_additional.append(lightcurves['Data'][i]['G_non']['mag'][G_non_mjd2_bool[i]])

            # Get only the last two non-detections - as long as they have come within timediff

            G_non_mjd_additional[i] = G_non_mjd_additional[i][-noofpoint:]
            G_non_mag_additional[i] = G_non_mag_additional[i][-noofpoint:]


            # The new data vectors are added to the existing vectors

            lightcurves['Data'][i]['G_mag_wn'] = np.append(G_non_mag_additional[i],lightcurves['Data'][i]['G_mag'])
            lightcurves['Data'][i]['G_mjd_wn'] = np.append(G_non_mjd_additional[i],lightcurves['Data'][i]['G_mjd'])

            # Remove NaNs
            lightcurves['Data'][i]['G_mag_wn'] = lightcurves['Data'][i]['G_mag_wn'][~np.isnan(lightcurves['Data'][i]['G_mag_wn'])] # Removes NaNs
            lightcurves['Data'][i]['G_mjd_wn'] = lightcurves['Data'][i]['G_mjd_wn'][~np.isnan(lightcurves['Data'][i]['G_mjd_wn'])]

            # A new error list is added as well, with magerror being put in
            lightcurves['Data'][i]['G_err_wn'] = np.append(np.repeat(-1,len(G_non_mag_additional[i])),lightcurves['Data'][i]['G_err'])
            lightcurves['Data'][i]['G_err_wn'] = lightcurves['Data'][i]['G_err_wn'][~np.isnan(lightcurves['Data'][i]['G_err_wn'])] # remove NaN again
            
    return lightcurves

