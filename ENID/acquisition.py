import os
import pickle
from alerce.core import Alerce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

token = '591d4e074bca6a265191593cf4c0e8609e488e1b'
alerce = Alerce()

def alerce_retrieval(obj_id):
    # INPUT
    # obj_id : object name (string, not list) for which a light curve must be obtained
    # OUTPUT
    # detections : 
    # non detections : 
    
    print('\nImporting lightcurve and metadata for ', '\033[1m' + obj_id + '\033[0m')
    
    # Import Object Information from Alerce
    lightcurve = alerce.query_lightcurve(obj_id, format="json")

    # Separate Detections and Non-detections
    non_detections = pd.DataFrame(lightcurve['non_detections'])
    detections = pd.DataFrame(lightcurve['detections'])

    # Define the columns that must be extracted from the dataframe (see pandas documentation for how to get columns)
    data_cols = ['mjd', 'magpsf', 'sigmapsf', 'fid']
    non_detection_cols = ['mjd', 'diffmaglim', 'fid']

    # Generate data and metadata data frames
    detections = detections[data_cols]
    
    if non_detections.empty == False:
        non_detectíons = non_detections[non_detection_cols]
    else:
        non_detections = -1
        
    return detections, non_detections

def source_search_alerce(obj_id):
    # INPUT
    # obj_id : list of object names to be queried in ALeRCE
    # OUTPUT
    # objects : source name and metadata for given obj_id candidates that were present in ALeRCE
    # missing : names of missing sources in ALeRCE, for which no data was found

    # Create empty dataframe with correct column names for indexing
    objects = pd.DataFrame(columns=['oid', 'ndethist', 'ncovhist', 'mjdstarthist', 'mjdendhist',
           'corrected', 'stellar', 'ndet', 'g_r_max', 'g_r_max_corr', 'g_r_mean',
           'g_r_mean_corr', 'firstmjd', 'lastmjd', 'deltajd', 'meanra', 'meandec',
           'sigmara', 'sigmadec', 'class', 'classifier', 'probability',
           'step_id_corr'])
    
    i = 0
    
    # Query ALeRCE database for objects corresponding to all elements in obj_id
    # This is done for 10 sources at a time to speed up the process
    while i < len(obj_id):
        query = alerce.query_objects(oid=obj_id[i:i+10], format="pandas")
        objects = objects.append(query.iloc[0:query.shape[0]])
        i += 10
                                  
    # Count the number of objects in obj_id for which nothing was found in ALeRCE
                                  
    # Also print the IDs for further investigation
    missing = list([])
                                  
    for object_ztf in obj_id:
        if object_ztf not in np.array([objects['oid']]):
            missing.append(object_ztf)
            
    return objects, missing

def alerce_class_query(class_type, num_detections, num_sources):
    # INPUT :
    # class_type : STRING - desired transient class to search for objects
    # num_detections : LIST of length 2 - lower and upper limit for number of detections of the source
    # num_sources : INT - number of maximal sources desired as output
    
    objects = alerce.query_objects(classifier="lc_classifier",
                           class_name=class_type, 
                           ndet=num_detections,
                           page_size=num_sources, format='pandas')

    objects_oid = objects['oid']
    
    print('\033[1m', 'Number of', class_type, 'sources : ', str(len(objects_oid)), '\033[0m')
    
    objects.head()
    
    return objects_oid

def dictionary(detections, non_detections):
    
    idx1 = np.where(detections['fid'] == 1)[0]
    idx2 = np.where(detections['fid'] == 2)[0]
    
    if type(non_detections) == int:
        lightcurve = {
            'R_mag': np.array(detections.loc[idx1, 'magpsf']),
            'R_err': np.array(detections.loc[idx1, 'sigmapsf']),
            'R_mjd': np.array(detections.loc[idx1, 'mjd']),
            'R_non': {'mag': non_detections, 'mjd': non_detections},
            'G_mag': np.array(detections.loc[idx2, 'magpsf']),
            'G_err': np.array(detections.loc[idx2, 'sigmapsf']),
            'G_mjd': np.array(detections.loc[idx2, 'mjd']),
            'G_non': {'mag': non_detections, 'mjd': non_detections}
        }
        
    else:
        idx1_non = np.where(non_detections['fid'] == 1)[0]
        idx2_non = np.where(non_detections['fid'] == 2)[0]
    
        lightcurve = {
            'R_mag': np.array(detections.loc[idx1, 'magpsf']),
            'R_err': np.array(detections.loc[idx1, 'sigmapsf']),
            'R_mjd': np.array(detections.loc[idx1, 'mjd']),
            'R_non': {'mag': np.array(non_detections.loc[idx1_non, 'diffmaglim']), 'mjd': np.array(non_detections.loc[idx1_non, 'mjd'])},
            'G_mag': np.array(detections.loc[idx2, 'magpsf']),
            'G_err': np.array(detections.loc[idx2, 'sigmapsf']),
            'G_mjd': np.array(detections.loc[idx2, 'mjd']),
            'G_non': {'mag': np.array(non_detections.loc[idx2_non, 'diffmaglim']), 'mjd': np.array(non_detections.loc[idx2_non, 'mjd'])}
        }
    
    return lightcurve

def lc_compile(object_name):
    
    detections, non_detections = alerce_retrieval(object_name)
    
    lightcurve = dictionary(detections, non_detections)

    return lightcurve

def ZTFandtype(csvfile):
    TNS = pd.read_csv(csvfile,skiprows=1)
    Typeextracts = pd.notna(TNS['type'])
    TNS = TNS[Typeextracts.values]
    
    TNS['internal_names'] = TNS['internal_names'] + ','
    ZTFsub = 'ZTF' + TNS['internal_names'].str.extract('ZTF(.+?),')
    ZTFextracts = pd.notna(ZTFsub)
    TNSZTF = TNS[ZTFextracts.values]
    ZTFsub = ZTFsub[ZTFextracts.values]
    ZTFsub = np.array(ZTFsub[0])
    ZTFsub = pd.DataFrame({'ZTF_names': ZTFsub})
    TNSZTF = TNSZTF.reset_index(drop = True)
    TNSZTF_gathered = pd.concat([TNSZTF, ZTFsub], axis=1)
    return TNSZTF_gathered

def dataset_gen(TNS_filename, pickle_filename):

    object_dictionary = {"Name": [], "Data": [], "Label": []}
    
    TNS_class = ZTFandtype(TNS_filename)
    TNS_internal_name = np.array(TNS_class['ZTF_names'])
    TNS_type = np.array(TNS_class['type'])
    TNS_typeid = np.array(TNS_class['typeid'])
    
    i = 0
    for object_name in TNS_internal_name:
        truth_check = alerce.query_objects(oid=object_name)
         
        if truth_check.shape[0] > 0:
            lightcurve = lc_compile(object_name) 
            
            if len(lightcurve['R_mag']) > 5 and len(lightcurve['G_mag']) > 5:
                object_dictionary['Data'].append(lightcurve)
                object_dictionary['Name'].append(TNS_internal_name[i])
                object_dictionary['Label'].append([TNS_type[i], TNS_typeid[i]])
            else:
                print('Warning : Not enought detections. Ignoring entry.')
        i += 1
    
    save_file = open(pickle_filename, mode='wb')
    pickle.dump(object_dictionary, save_file)
    save_file.close()

    return object_dictionary, pickle_filename
