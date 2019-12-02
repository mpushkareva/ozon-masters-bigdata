#!/opt/conda/envs/dsenv/bin/python

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

import os
import sys

SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.7-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

from pyspark import SparkConf

conf = SparkConf()
conf.set("spark.jars.packages", "graphframes:graphframes:0.7.0-spark2.3-s_2.11")

spark = SparkSession.builder.config(conf=conf).getOrCreate()

from pyspark.sql.types import *

import pyspark.sql.functions as f

def shortest_path(v_from, v_to, df_name, output, max_path_length=10):
    
    schema = StructType(fields=[
    StructField("user_id", StringType()),
    StructField("follower_id", StringType())])
    
    df = spark.read.schema(schema).format("csv").option("sep", "\t").load(df_name)
    
    df_sel = df.where(df.follower_id == v_from)
    df_paths = df_sel.select(f.concat_ws(",",  "follower_id", "user_id").alias("path"), df_sel.user_id.alias("next"))
    for i in range(max_path_length):
        if df_paths.where(df_paths.next == v_to).count() == 0:
            df_ext = df_paths.join(df.select(df.follower_id.alias("next"), df.user_id), on="next", how="inner")
            df_paths = df_ext.select(f.concat_ws(",", "path",  "user_id").alias("path"), df_ext.user_id.alias("next"))
        else: df_paths.select("path").where(df_paths.next == v_to).write.mode("overwrite").text(output)
    spark.stop()

shortest_path(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

