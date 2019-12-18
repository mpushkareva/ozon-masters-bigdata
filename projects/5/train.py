#!/opt/conda/envs/dsenv/bin/python

import os
import sys

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

from pyspark.sql.types import *
from pyspark.ml.feature import *
import pyspark.sql.functions as f
from pyspark.ml import Pipeline
from model import pipeline, sklearn_est
import joblib import dump
from sklearn.linear_model import LogisticRegression
import pickle

path = sys.argv[1]
train =  spark.read.json(path)
pipeline_model = pipeline.fit(train)

pipeline_model = pipeline.fit(train)
train_transformed = pipeline_model.transform(train)

est = LogisticRegression(random_state=5757)
est_broadcast = spark.sparkContext.broadcast(est)

@F.udf(ArrayType(FloatType()))
def vectorToArray(row):
    return row.tolist()

@F.pandas_udf(FloatType())
def predict(series):
    predictions = est_broadcast.value.predict(series)
    return pd.Series(predictions)

with open("logistic_model.pk", "wb") as f:
    pickle.dump(est, f)
    
spark_est = SKLogisticRegreesionModel(model_file="logistic_model.pk", featuresCol="features",\
                                      vectorToArray=vectorToArray, predict=predict)
##spark_est.transform(df_test)


dump(spark_est, "{}.joblib.format(5))
