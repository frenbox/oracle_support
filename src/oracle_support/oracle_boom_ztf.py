#load json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import torch
torch.set_default_device("cpu")
import astropy


import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord

from oracle.custom_datasets.BTS import ZTF_passband_to_wavelengths
from oracle.custom_datasets.BTS import time_dependent_feature_list, time_independent_feature_list, meta_data_feature_list, flag_value

from oracle.presets import get_model

path_aux = Path("../../data/alert_aux.json")

path_alert = Path("../../data/alert.json")
prv_cand = None

# Loading the model
Oracle2 = get_model("BTSv2")
Oracle2.load_state_dict(torch.load("../../data/best_model_f1.pth", map_location='cpu'), strict=False)
Oracle2.eval()

with open(path_aux, "r") as f:
    data = json.load(f)
    prv_cand = data['prv_candidates']
    cross_matches = data['cross_matches']
    prv_cand = pd.DataFrame(prv_cand)
    prv_cand.sort_values('jd', inplace=True)    

with open(path_alert, "r") as f:
    data = json.load(f)
    candidate = data['candidate']



# convert the jd to time since first detection 
prv_cand['jd'] = prv_cand['jd'] - prv_cand['jd'].min()

# convert the filter ids to mean wavelengths
prv_cand['band'] = prv_cand['band'].map(ZTF_passband_to_wavelengths)

# we use galactic coordinates as static features, so convert the ra and dec to l and b
coords = SkyCoord(ra=prv_cand['ra'].to_numpy()*u.deg, dec=prv_cand['dec'].to_numpy()*u.deg, frame='icrs')
prv_cand['l'] = coords.galactic.l
prv_cand['b'] = coords.galactic.b

# Add the wise colors for nearest source within 2" 
prv_cand['W1mag']       = cross_matches['AllWISE'][0]['w1mpro']
prv_cand['W2mag']       = cross_matches['AllWISE'][0]['w2mpro']
prv_cand['W3mag']       = cross_matches['AllWISE'][0]['w3mpro']
prv_cand['W4mag']       = cross_matches['AllWISE'][0]['w4mpro']
prv_cand['W1_minus_W3'] = cross_matches['AllWISE'][0]['w1mpro'] - cross_matches['AllWISE'][0]['w3mpro']
prv_cand['W2_minus_W3'] = cross_matches['AllWISE'][0]['w2mpro'] - cross_matches['AllWISE'][0]['w3mpro'] 

prv_cand['sgscore1']    =  candidate['sgscore1'] 
prv_cand['sgscore2']    =  candidate['sgscore2']
prv_cand['distpsnr1']   =  candidate['distpsnr1'] 
prv_cand['distpsnr2']   =  candidate['distpsnr2']
prv_cand['ndethist']    =  candidate['ndethist']
prv_cand['nmtchps']     =  candidate['nmtchps']
prv_cand['drb']         =  candidate['drb']
prv_cand['ncovhist']    =  candidate['ncovhist'] 
prv_cand['sgmag1']      =  candidate['sgmag1'] 
prv_cand['srmag1']      =  candidate['srmag1']
prv_cand['simag1']      =  candidate['simag1']
prv_cand['szmag1']      =  candidate['szmag1'] 
prv_cand['sgmag2']      =  candidate['sgmag2']
prv_cand['srmag2']      =  candidate['srmag2'] 
prv_cand['simag2']      =  candidate['simag2']
prv_cand['szmag2']      =  candidate['szmag2'] 

# 1 is the batch size
ts_tensor = torch.zeros((1, len(prv_cand), len(time_dependent_feature_list) + 1))
for i, col in enumerate(time_dependent_feature_list):
    ts_tensor[0, :, i] = torch.tensor(prv_cand[col].values)

static_tensor = torch.zeros((1, len(time_independent_feature_list)))
for i, col in enumerate(time_independent_feature_list):
    try:
        static_tensor[0, i] = torch.tensor(prv_cand[col].values[-1])  # Use the last value for static features
    except KeyError:
        print(f"Column {col} not found in data. Setting to default flag value.")
        static_tensor[0, i] = flag_value  # Default value if no data is available

meta_data_tensor = torch.zeros((1, len(meta_data_feature_list)))
for i, col in enumerate(meta_data_feature_list):
    try:
        meta_data_tensor[0, i] = torch.tensor(prv_cand[col].values[-1])  # Use the last value for meta data features
    except KeyError:
        print(f"Column {col} not found in data. Setting to default flag value.")
        meta_data_tensor[0, i] = flag_value  # Default value if no data is available

length = torch.tensor([len(prv_cand)])
static_tensor = torch.cat((static_tensor, meta_data_tensor), dim=1)

batch = {
    'ts': ts_tensor,
    'static': static_tensor,
    'length': length,
}

with torch.no_grad():

    class_scores = Oracle2.predict_class_probabilities(batch)[0]
    class_scores_df = Oracle2.predict_conditional_probabilities_df(batch)
    # Oracle2.taxonomy.plot_colored_taxonomy(class_scores)

    print(class_scores_df)