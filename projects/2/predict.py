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
        df = pd.DataFrame([id, if1, if2, if3, if4, if5,
        if6, if7, if8, if9, if10, if11, if12,
        if13, if14, cf2, cf3, cf4])
        pred = model.predict(df)
        out = zip(df[0], pred)
        print'\t'.join(map(str, [fname, age]))
        print("\n".join(["{0}\t{1}".format(*i) for i in out]))
