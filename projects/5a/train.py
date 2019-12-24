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

global spark
spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')
conf = SparkConf()

from sklearn import linear_model
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
from pickle import loads, dumps
import base64


from pyspark.sql.types import *
from pyspark.ml.feature import *
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
#from model import pipeline, sklearn_est
from joblib import dump
from sklearn.linear_model import LogisticRegression
import pickle
from pyspark.ml import PipelineModel
from sklearn_wrapper import SklearnEstimatorModel


    
stop_words = StopWordsRemover.loadDefaultStopWords("english")
tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="wordsReview", pattern="\\W")
swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="reviewFiltered", stopWords=stop_words)
count_vectorizer = CountVectorizer(inputCol=swr.getOutputCol(), outputCol="reviewVector", binary=True, vocabSize=20)
assembler = VectorAssembler(inputCols=[count_vectorizer.getOutputCol(), 'verified'], outputCol="features")



@F.udf(ArrayType(DoubleType()))
def vectorToArray(row):
    return row.toArray().tolist()

@F.pandas_udf(DoubleType())
def predict(series):
    predictions = est_broadcast.value.predict(series.tolist())
    return pd.Series(predictions)

class HasSklearnModel(Params):
    sklearn_model = Param(Params._dummy(), "sklearn_model", "sklearn_model",
        typeConverter=TypeConverters.toString)
    
    def init(self):
        super(HasSklearnModel, self).init()
        kwargs = self._input_kwargs
        self._set(**kwargs)
        
    def setSklearnModel(self, value):
        return self._set(sklearn_model=value)

    def getSklearnModel(self):
        return self.getOrDefault(self.sklearn_model)
class SklearnEstimatorModel(Model, HasSklearnModel, HasFeaturesCol, HasPredictionCol,
                           DefaultParamsReadable, DefaultParamsWritable):
    #sklearn_model = Param(Params._dummy(), "sklearn_model", "sklearn_model",
     #   typeConverter=TypeConverters.toString)
    
    @keyword_only
    def __init__(self, sklearn_model=None, featuresCol="features", predictionCol="prediction"):
        super(SklearnEstimatorModel, self).__init__()
        if sklearn_model is None:
            raise ValueError("model_file must be specified!")
        #with open(model_file, "rb") as f:
        #    self.estimator = load(model_file)
        self.setSklearnModel(sklearn_model)
        self.est = loads(base64.b64decode(self.getSklearnModel().encode('utf-8')))
        #self.spark = spark
        kwargs = self._input_kwargs
        self._set(**kwargs)
        
    def _transform(self, dataset):
        global est_broadcast
        #global spark
        est_broadcast = spark.sparkContext.broadcast(self.est)
        dataset = dataset.withColumn("features_array", vectorToArray("features")).localCheckpoint()
        return dataset.withColumn("prediction", predict("features_array"))
    
class SklearnEstimator(Estimator, HasSklearnModel, HasFeaturesCol, HasPredictionCol, HasLabelCol,
                           DefaultParamsReadable, DefaultParamsWritable):
    @keyword_only
    def __init__(self, featuresCol="features", predictionCol="prediction", labelCol="label"):
        super(SklearnEstimator, self).__init__()
        kwargs = self._input_kwargs
        #self.spark = spark
        self._set(**kwargs)
        
    def _fit(self, dataset):
        local_dataset = dataset.select(self.getFeaturesCol(), self.getLabelCol()).toPandas()
        self.est = LogisticRegression()
        #with open(self.model_file, "wb") as f:
         #   pickle.dump(self.est, f)    
        self.est.fit(local_dataset[self.getFeaturesCol()].tolist(), local_dataset[self.getLabelCol()])
        model_string = base64.b64encode(dumps(self.est)).decode('utf-8')
        self.setSklearnModel(model_string)
        return SklearnEstimatorModel(sklearn_model=model_string, predictionCol=self.getPredictionCol(),
                                         featuresCol=self.getFeaturesCol())
spark_est = SklearnEstimator()
pipeline = Pipeline(stages=[
    tokenizer,   
    swr,    
    count_vectorizer, 
    assembler, 
    spark_est
])
path = sys.argv[1]
train =  spark.read.json(path)

pipeline_model = pipeline.fit(train)
train_transformed = pipeline_model.transform(train)
pipeline_model.write().overwrite().save(sys.argv[2])

local_dataset = train_transformed.select("features", "label").toPandas()
est = LogisticRegression(random_state=5757)
est.fit(local_dataset["features"].tolist(), local_dataset["label"])
dump(est, sys.argv[3])

#with open("logistic_model.pk", "wb") as f:
#    pickle.dump(est, f)
    
#spark_est = SklearnEstimatorModel(model_file="logistic_model.pk", featuresCol="features", labelCol="label")
##spark_est.transform(df_test)


#dump(spark_est, "{}.joblib".format(sys.argv[3]))
