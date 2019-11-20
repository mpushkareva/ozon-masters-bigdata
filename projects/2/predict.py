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

read_opts=dict(
        sep='\s', names=fields_without_label, index_col=False, header=None,
        iterator=True, chunksize=1000
)
fields_selected = ["id"] + ["if"+str(i) for i in range(1,14)] \
                + ["cf2", "cf3", "cf4"]

for line in sys.stdin:
        id, if1, if2, if3, if4, if5,\
        if6, if7, if8, if9, if10, if11, if12,\
        if13, if14, cf2, cf3, cf4 = line.strip('\n').split('\s')
        if if1 = '\\N': if1 = '0'
        if if2 = '\\N': if2 = '0'
        if if3 = '\\N': if3 = '0'
        if if4 = '\\N': if4 = '0'
        if if5 = '\\N': if5 = '0'
        if if6 = '\\N': if6 = '0'
        if if7 = '\\N': if7 = '0'
        if if8 = '\\N': if8 = '0'
        if if9 = '\\N': if9 = '0'
        if if10 = '\\N': if10 = '0'
        if if11 = '\\N': if11 = '0'
        if if12 = '\\N': if12 = '0'
        if if13 = '\\N': if13 = '0'
        if if14 = '\\N': if14 = '0'
        if cf2 = '\\N': cf2 =''
        if cf3 = '\\N': cf3 =''
        if cf4 = '\\N': cf4 =''
        df = pd.DataFrame([id, if1, if2, if3, if4, if5,
        if6, if7, if8, if9, if10, if11, if12,
        if13, if14, cf2, cf3, cf4])
        pred = model.predict(df)
        out = zip(df[0], pred)
        print("{0}\s{1}".format(*i) for i in out))
