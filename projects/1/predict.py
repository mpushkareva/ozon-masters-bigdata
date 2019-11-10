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
model = load("1.joblib")

#fields = """doc_id,hotel_name,hotel_url,street,city,state,country,zip,class,price,
#num_reviews,CLEANLINESS,ROOM,SERVICE,LOCATION,VALUE,COMFORT,overall_ratingsource""".replace("\n",'').split(",")

#read and infere
numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)] + ["day_number"]

fields_without_label = ["id"] + numeric_features + categorical_features
read_opts=dict(
        sep='\s', names=fields_without_label, index_col=False, header=None,
        iterator=True, chunksize=200
)
fields_selected = ["id"] + ["if"+str(i) for i in range(1,14)] \
                + ["cf1", "cf2", "cf3", "cf4"]
for df in pd.read_csv(sys.stdin, **read_opts):
    df_selected = df.loc[:, fields_selected]
    pred = model.predict(df_selected)
    out = zip(df_selected.doc_id, pred)
    print("\n".join(["{0}\t{1}".format(*i) for i in out]))
