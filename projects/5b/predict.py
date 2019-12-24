#!/opt/conda/envs/dsenv/bin/python

import os, sys

SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.7-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

from pyspark.sql import SparkSession
from pyspark import SparkConf

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')
conf = SparkConf()

import sklearn
import numpy as np
import pandas as pd
import sys
import json
import re
import base64
import pickle

from sklearn import linear_model
from sklearn.preprocessing import normalize
from pyspark import SparkConf
from pyspark.sql import SparkSession
from sklearn.feature_extraction.text import CountVectorizer
from pyspark.ml.feature import *
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from pyspark.ml import Estimator
from pyspark.ml import Pipeline
from pyspark.ml.linalg import DenseVector
from pyspark import keyword_only
from pyspark.ml import Model
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol, HasPredictionCol
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

model = PipelineModel.load(sys.argv[1])
test = spark.read.load(sys.argv[2], format="json")

predictions = model.transform(test)

predictions.write.format("json").save(sys.argv[3])

