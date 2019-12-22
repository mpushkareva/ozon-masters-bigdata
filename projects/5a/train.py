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
from joblib import dump
from sklearn.linear_model import LogisticRegression
import pickle
from pyspark.ml import PipelineModel

path = sys.argv[1]
train =  spark.read.json(path)

pipeline_model = pipeline.fit(train)
train_transformed = pipeline_model.transform(train)
pipeline_model.write().overwrite().save(sys.argv[2])

local_dataset = train_transformed.select("features", "label").toPandas()
est = LogisticRegression(random_state=5757)
est.fit(local_dataset["features"].tolist(), local_dataset["label"])
est_broadcast = spark.sparkContext.broadcast(est)
dump(est, "5.joblib")

#with open("logistic_model.pk", "wb") as f:
#    pickle.dump(est, f)
    
#spark_est = SklearnEstimatorModel(model_file="logistic_model.pk", featuresCol="features", labelCol="label")
##spark_est.transform(df_test)


#dump(spark_est, "{}.joblib".format(sys.argv[3]))
