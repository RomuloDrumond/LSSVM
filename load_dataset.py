# loading datasets
import pandas as pd

# Vertebral Column [http://archive.ics.uci.edu/ml/datasets/vertebral+column]
# dataset for classification between Normal (NO) and Abnormal (AB)
vc2c = pd.read_csv('../datasets/vertebral_column/column_2C.dat', delim_whitespace=True, header=None)
# dataset for classification between DH (Disk Hernia), Spondylolisthesis (SL) and Normal (NO)
vc3c = pd.read_csv('../datasets/vertebral_column/column_3C.dat', delim_whitespace=True, header=None)

# Wall-Following [https://archive.ics.uci.edu/ml/datasets/Wall-Following+Robot+Navigation+Data]
# dataset with all 24 ultrassound sensors readings
wf24f = pd.read_csv('../datasets/wall_following/sensor_readings_24.data', header=None)
# dataset with simplified 4 readings (front, left, right and back)
wf4f  = pd.read_csv('../datasets/wall_following/sensor_readings_4.data',  header=None)
# dataset with simplified 2 readings (front and left)
wf2f  = pd.read_csv('../datasets/wall_following/sensor_readings_2.data',  header=None)

# Parkinson [https://archive.ics.uci.edu/ml/datasets/parkinsons]
# (31 people, 23 with Parkinson's disease (PD))
temp = pd.read_csv('../datasets/parkinson/parkinsons.data')
labels = temp.columns.values.tolist()
new_labels = [label for label in labels if label not in ('name')] # taking off column 'name'
pk = temp[new_labels]


pk_features = pk.columns.tolist()
pk_features.remove('status')

# datasets with separation between 'features' and 'labels'
datasets = {
    "vc2c":  {"features": vc2c.iloc[:,0:6],  "labels": pd.get_dummies(vc2c.iloc[:,6],  drop_first=True)},
    "vc3c":  {"features": vc3c.iloc[:,0:6],  "labels": pd.get_dummies(vc3c.iloc[:,6],  drop_first=True)},
    "wf24f": {"features": wf24f.iloc[:,0:24],"labels": pd.get_dummies(wf24f.iloc[:,24],drop_first=True)},
    "wf4f":  {"features": wf4f.iloc[:,0:4],  "labels": pd.get_dummies(wf4f.iloc[:,4],  drop_first=True)},
    "wf2f":  {"features": wf2f.iloc[:,0:2],  "labels": pd.get_dummies(wf2f.iloc[:,2],  drop_first=True)},
    "pk":    {"features": pk.loc[:,pk_features], "labels": pk.loc[:,["status"]]}
}

'''
OBS: Was chosen to maintain k-1 dummies variables when we had k categories, so the missing category is identified when all dummies variables are zero.
'''

import numpy as np

# printing datasets info
print("{:10}{:18}{:}".format(
        'Dataset:',
        'Features.shape:',
        '# of classes:',
        ))
for dataset_name, data in datasets.items():
    print("{:9} {:17} {:}".format(
        dataset_name, 
        str(data['features'].shape),
        len(np.unique(data['labels'].values, axis=0))
        ))