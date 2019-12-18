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

from pyspark.ml import Pipeline, PipelineModel
from sklearn_wrapper import SklearnEstimatorModel

model = PipelineModel.load(sys.argv[1])
test_path = sys.argv[3]
test =  spark.read.json(test_path)
test_transformed = model.transform(test)
spark_est = SklearnEstimatorModel(model_path=arg[2])

predictions = spark_est.transform(test_transformed)

predictions.select("id","prediction").write.parquet(sys.argv[4], mode="overwrite")





