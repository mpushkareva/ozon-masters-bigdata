#!/opt/conda/envs/dsenv/bin/python

import os, sys
'''
SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.7-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

from pyspark.sql import SparkSession
from pyspark import SparkConf

global spark
spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')
conf = SparkConf()
'''
from pyspark.sql import SparkSession

#global spark
spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from pyspark import keyword_only
from pyspark.ml import Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol, HasPredictionCol
from pyspark.sql.types import *
from pyspark.ml.feature import *
import pyspark.sql.functions as F
#from pyspark.ml import Estimator
from sklearn.linear_model import LogisticRegression
import pandas as pd
from joblib import load, dump
from pyspark.ml import Estimator
from pickle import loads, dumps
import base64
from pyspark.ml import Pipeline, PipelineModel
import model

from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

model1 = PipelineModel.load(sys.argv[1])
test = spark.read.load(sys.argv[2], format="json")

predictions = model1.transform(test)

predictions.write.format("json").save(sys.argv[3])

