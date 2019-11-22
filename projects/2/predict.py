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
        sep='\t', names=fields_without_label, index_col=False, header=None,
        iterator=True, chunksize=1000
)
fields_selected = ["id"] + ["if"+str(i) for i in range(1,14)] \
                + ["cf2", "cf3", "cf4"]

for line in sys.stdin:
        df = pd.DataFrame(line.strip('\n').split('\t'))
        df.iloc[:,0:15].replace('\\N', '0')
        df.iloc[:,0:15] = df.iloc[:,0:15].astype(int)
        pred = model.predict(df)
        out = zip(df[0][0], pred)
        print("\n".join(["{0}\t{1}".format(*i) for i in out]))
