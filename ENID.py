import os
import json
import requests

from collections import OrderedDict
from lasair import LasairError, lasair_client as lasair
from alerce.core import Alerce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import time

token = '591d4e074bca6a265191593cf4c0e8609e488e1b'
L = lasair(token)
alerce = Alerce()

url_tns_api = 0
TNS_BOT_ID = 0
TNS_BOT_NAME = 0

def set_var(var1, var2, var3, var4):
    
    global url_tns_api
    url_tns_api = var1
    
    global TNS_BOT_ID 
    TNS_BOT_ID = var2
    
    global TNS_BOT_NAME 
    TNS_BOT_NAME = var3

    global TNS_API_KEY
    TNS_API_KEY = var4

def set_bot_tns_marker():
    tns_marker = 'tns_marker{"tns_id": "' + str(TNS_BOT_ID) + '", "type": "bot", "name": "' + TNS_BOT_NAME + '"}'
    return tns_marker

def format_to_json(source):
    parsed = json.loads(source, object_pairs_hook = OrderedDict)
    result = json.dumps(parsed, indent = 4)
    return result

def is_string_json(string):
    try:
        json_object = json.loads(string)
    except Exception:
        return False
    return json_object

def print_status_code(response):
    json_string = is_string_json(response.text)
    if json_string != False:
        print ("status code ---> [ " + str(json_string['id_code']) + " - '" + str(json_string['id_message']) + "' ]\n")
    else:
        status_code = response.status_code
        if status_code == 200:
            status_msg = 'OK'
        elif status_code in ext_http_errors:
            status_msg = err_msg[ext_http_errors.index(status_code)]
        else:
            status_msg = 'Undocumented error'
        print ("status code ---> [ " + str(status_code) + " - '" + status_msg + "' ]\n")

def search(search_obj):
    search_url = url_tns_api + "/search"
    tns_marker = set_bot_tns_marker()
    headers = {'User-Agent': tns_marker}
    json_file = OrderedDict(search_obj)
    search_data = {'api_key': TNS_API_KEY, 'data': json.dumps(json_file)}
    response = requests.post(search_url, headers = headers, data = search_data)
    return response

def get(get_obj):
    get_url = url_tns_api + "/object"
    tns_marker = set_bot_tns_marker()
    headers = {'User-Agent': tns_marker}
    json_file = OrderedDict(get_obj)
    get_data = {'api_key': TNS_API_KEY, 'data': json.dumps(json_file)}
    response = requests.post(get_url, headers = headers, data = get_data)
    return response

def get_file():
    filename = os.path.basename(file_tns_url)
    tns_marker = set_bot_tns_marker()
    headers = {'User-Agent': tns_marker}
    api_data = {'api_key': TNS_API_KEY}
    print ("Downloading file '" + filename + "' from the TNS...\n")
    response = requests.post(file_tns_url, headers = headers, data = api_data, stream = True)    
    print_status_code(response)
    path = os.path.join(download_dir, filename)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in response:
                f.write(chunk)
        print ("File was successfully downloaded.\n")
    else:
        print ("File was not downloaded.\n")

def print_response(response, json_file, counter):
    response_code = str(response.status_code) if json_file == False else str(json_file['id_code'])
    stats = 'Test #' + str(counter) + '| return code: ' + response_code + \
            ' | Total Rate-Limit: ' + str(response.headers.get('x-rate-limit-limit')) + \
            ' | Remaining: ' + str(response.headers.get('x-rate-limit-remaining')) + \
            ' | Reset: ' + str(response.headers.get('x-rate-limit-reset'))
    if(response.headers.get('x-cone-rate-limit-limit') != None):
        stats += ' || Cone Rate-Limit: ' + str(response.headers.get('x-cone-rate-limit-limit')) + \
                 ' | Cone Remaining: ' + str(response.headers.get('x-cone-rate-limit-remaining')) + \
                 ' | Cone Reset: ' + str(response.headers.get('x-cone-rate-limit-reset'))
    print (stats)

def get_reset_time(response):
    # If any of the '...-remaining' values is zero, return the reset time
    for name in response.headers:
        value = response.headers.get(name)
        if name.endswith('-remaining') and value == '0':
            return int(response.headers.get(name.replace('remaining', 'reset')))
    return None

def rate_limit_handling(search_obj):
    counter = 0
    while True:
        counter = counter + 1
        response = search(search_obj)
        json_file = is_string_json(response.text)
        print_response(response, json_file, counter)
        # Checking if rate-limit reached (...-remaining = 0)
        reset = get_reset_time(response)
        # A general verification if not some error 
        if (response.status_code == 200):
            if reset != None:
                # Sleeping for reset + 1 sec
                print("Sleep for " + str(reset + 1) + " sec") 
                time.sleep(reset + 1)
        	    # Can continue to submit requests...
                print ("Continue to submit requests...")
                for i in range(3):
                    counter = counter + 1
                    response = search()
                    json_file = is_string_json(response.text)
                    print_response(response, json_file, counter)
                print ("etc...\n") 
                break
        else:
            print_status_code(response)       
            break
           
def alerce_retrieval(obj_id):
    # INPUT
    # obj_id : object name (string, not list) for which a light curve must be obtained
    # OUTPUT
    # data : light curve data. Dataframe consisting of 4 columns : date and magnitude of source in both G and R bands
    # metadata : metadata for given source (dataframe)
    
    obj_detection, obj_nondetection = source_search_alerce([obj_id])
    
    if obj_detection['ndet'][0] < 5:
        data = None
        metadata = None
	
    else:
        print('Importing lightcurve and metadata for source ', '\033[1m' + obj_id + '\033[0m')
    
        # Import Object Information from Alerce
        lightcurve = alerce.query_lightcurve(obj_id, format="json")
    
        # Separate Detections and Non-detections
        non_detections = pd.DataFrame(lightcurve['non_detections'])
        detections = pd.DataFrame(lightcurve['detections'])
    
        # Define the columns that must be extracted from the dataframe (see pandas documentation for how to get columns)
        data_cols = ['mjd', 'magpsf']
        metadata_cols = ['fid', 'ra', 'dec']
    
        # Generate data and metadata data frames
        data_unsorted = detections[data_cols]
        metadata = detections[metadata_cols]
    
        # Retrieve indexes for the two filter IDs
        idx1 = np.where(np.array([metadata['fid']])[0] == 1)[0]
        idx2 = np.where(np.array([metadata['fid']])[0] == 2)[0]
    
        g_band = data_unsorted.loc[idx1, data_cols]
        r_band = data_unsorted.loc[idx2, data_cols]
    
        # Generate final dataframe and rename columns to distinguish the two filters
        data = pd.concat([g_band, r_band], axis=1)
        data.columns = ['mjd_g','magpsf_g','mjd_r','magpsf_r']
    
    return data, metadata

def source_search_lasair(obj_id):
    # INPUT
    # obj_id : list of object names to be queried in Lasair-Iris
    # OUTPUT
    # objects : source name and metadata for given obj_id candidates that were present in Lasair-Iris
    # missing : names of missing sources in Lasair-Iris, for which no data was found
    
    print('\nSearching for missing objects on Lasair-Iris...')
    
    # Query Lasair-Iris for the objects that were not found in ALeRCE
    lasair_obj = L.objects(obj_id)

    lasair_found = []
    lasair_missing = []
    
    # Determine whether there is any data for the sources that were queried
    for i in range(len(obj_id)):
        # check whether the output from the L.objects command is empty, in which case no data can be found
        if not lasair_obj[i]:  
            lasair_missing.append(obj_id[i])
        else:
            lasair_found.append(obj_id[i])
    
    # Count the number of objects that were found on Lasair-Iris
    print('Found', len(lasair_found), 'objects.\n')
    for candidate in lasair_found:
        print(candidate)
    
    # Count and display the objects that are still missing
    print('\n' + str(len(lasair_missing)), 'objects not retrievable.\n')
    for candidate in lasair_missing:
        print(candidate)
        
    return lasair_found, lasair_missing

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

def lightcurve_plot(data, source):
    # INPUT : 
    # data : light curve information, in the same dataframe format as obtained from alerce_retrieval
    # source : source name
                                  
    # Plot the lightcurve for the input source in both G and R bands and display source name
    fig = plt.figure()
    plt.scatter(data['mjd_g'], data['magpsf_g'], c='green', label='G-band')
    plt.scatter(data['mjd_r'], data['magpsf_r'], c='red', label='R-band')
    plt.gca().invert_yaxis()
    plt.title(source, fontweight="bold")
    plt.xlabel('Time [MJD]')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid()
    plt.show()
                                  
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

def TNS_get_class(object_id, TNS_objects):
    
    print('Querying TNS for', object_id,'name and class verification...') 
    
    TNS_ZTFobjects = TNS_objects['internal_names']
    TNS_objects = TNS_objects.loc[TNS_ZTFobjects == object_id]
    print(TNS_objects)
    TNS_class = TNS_objects['type']
    TNS_class_id = TNS_objects['typeid']
    
    print('TNS Class : ', TNS_class)
        
    return TNS_class, TNS_class_id

def dataset_gen(classes):
    
    lightcurves = pd.DataFrame()
    timestamps = pd.DataFrame()
    labels = pd.DataFrame()
    
    TNS_objects = pd.read_csv('tns_public_objects.csv', skiprows=1)
    
    for transient_class in classes:
        transient_objects = alerce_class_query(class_type = transient_class, num_detections=[10, 200], num_sources=10000)            
        i = 0
        
        for object_name in transient_objects:
            object_data, object_metadata = alerce_retrieval(object_name)
            if object_data is None:
                print('Not enough detections for \033[1m', object_name, '\033[0m - Entry ignored')
                continue
            
            else:
                TNS_class, TNS_label = TNS_get_class(object_name, TNS_objects)
            
                lightcurves = pd.concat([lightcurves, object_data['magpsf_r']])
                lightcurves = pd.concat([lightcurves, object_data['magpsf_g']])
                timestamps = pd.concat([timestamps, object_data['mjd_r']])
                timestamps = pd.concat([timestamps, object_data['mjd_g']])
                labels = pd.concat([labels, pd.DataFrame({'Class': [TNS_class], 'Class ID': [TNS_label], 'Filter ID': [1], 'Object Name': [object_name]})])
                labels = pd.concat([labels, pd.DataFrame({'Class': [TNS_class], 'Class ID': [TNS_label], 'Filter ID': [2], 'Object Name': [object_name]})])

                # For DEMO, remove afterwards
                i += 1
                if i > 4:
                    break
                
    lightcurves.set_index([np.arange(lightcurves.shape[0])], inplace=True)
    labels.set_index([np.arange(labels.shape[0])], inplace=True)
    timestamps.set_index([np.arange(timestamps.shape[0])], inplace=True)
    
    return lightcurves, labels, timestamps
