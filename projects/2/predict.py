#!/opt/conda/envs/dsenv/bin/python


import sys, os
import logging
from joblib import load
import pandas as pd

sys.path.append('.')

#load the model
model = load("2.joblib")


#read and infere
numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)] + ["day_number"]

fields_without_label = ["id"] + numeric_features + categorical_features
fields_selected = ["id"] + ["if"+str(i) for i in range(1,14)] 
read_opts=dict(
        sep='\t', names=fields_without_label, index_col=False, header=None,
        iterator=True, chunksize=1000
)

for df in pd.read_csv(sys.stdin, **read_opts):
    df_selected = df.loc[:, fields_selected]
    df_selected = df_selected.replace("\\N", "0")
    df_selected = df_selected.apply(pd.to_numeric)
    #df_selected.iloc[:, 14:17] = df_selected.iloc[:, 14:17].replace("\\N", "")
    pred = model.predict(df_selected)
    out = zip(df_selected.id, pred)
    print("\n".join(["{0}\t{1}".format(*i) for i in out]))
