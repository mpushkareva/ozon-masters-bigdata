#!/opt/conda/envs/dsenv/bin/python

import sys, os
import logging
from joblib import load
import pandas as pd

sys.path.append('.')
from model import fields

#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#load the model
model = load("2.joblib")


#read and infere
numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)] + ["day_number"]

fields_without_label = ["id"] + numeric_features + categorical_features
fields_selected = ["id"] + ["if"+str(i) for i in range(1,14)] \
                + ["cf2", "cf3", "cf4"]
read_opts=dict(
        sep='\t', names=fields_without_label, index_col=False, header=None,
        iterator=True, chunksize=1000
)

for df in pd.read_csv(sys.stdin, **read_opts):
    df_selected = df.loc[fields_selected]
    df_selected.iloc[:, 0:14] = df_selected.iloc[:, 0:14].replace('\\N', '0')
    df_selected.iloc[:, 0:14] = pd.to_numeric(df_selected.iloc[:, 0:14])
    pred = model.predict(df_selected)
    out = zip(df_selected.id, pred)
    print("\n".join(["{0}\t{1}".format(*i) for i in out]))
