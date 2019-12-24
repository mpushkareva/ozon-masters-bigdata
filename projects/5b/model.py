#!/opt/conda/envs/dsenv/bin/python

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
from joblib import dump

@F.pandas_udf(DoubleType())
def predict(series):
    # Необходимо сделать преобразования, потому что на вход приходит pd.Series(list)
    print(est_broadcast.value)
    predictions = est_broadcast.value.predict(series.tolist())
    return pd.Series(predictions)

@F.udf(ArrayType(DoubleType()))
def _sparseVectorToArray(row):
    return row.toArray().tolist()


class HasSklearnModel(Params):

    sklearn_model = Param(Params._dummy(), "sklearn_model", "sklearn_model",
        typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasSklearnModel, self).__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)
        
    def setSklearnModel(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(sklearn_model=value)

    def getSklearnModel(self):
        """
        Gets the value of outputCol or its default value.
        """
        return self.getOrDefault(self.sklearn_model)
    
class SklearnEstimatorModel(Model, HasFeaturesCol, HasLabelCol, HasPredictionCol, HasSklearnModel,
                           DefaultParamsReadable, DefaultParamsWritable,):
    sklearn_model = Param(Params._dummy(), "sklearn_model",
                      "path to pickled scikit-learn logistic regression model",
                      typeConverter=TypeConverters.toString)
    @keyword_only
    def __init__(self, sklearn_model=None, featuresCol="features", labelCol="label", 
                 predictionCol="prediction"):
        super(SklearnEstimatorModel, self).__init__()   
        self.setSklearnModel(sklearn_model)
        kwargs = self._input_kwargs
        self._set(**kwargs)


    @keyword_only
    def setParams(self, sklearn_model=None, featuresCol="features", labelCol="label", 
                 predictionCol="prediction"):
        self.setSklearnModel(sklearn_model)
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setValue(self, value):
        return self._set(value=value)

    def getValue(self):
        return self.getOrDefault(self.value)
        
    def _transform(self, dataset):
        self.estimator = pickle.loads(base64.b64decode(self.getSklearnModel().encode('utf-8')))
        global est_broadcast
        est_broadcast = spark.sparkContext.broadcast(self.estimator)
        local_dataset = dataset.withColumn(self.getFeaturesCol(), _sparseVectorToArray("word_vector")).localCheckpoint()
        return local_dataset.withColumn(self.getPredictionCol(), predict(self.getFeaturesCol()))

class SklearnEstimator(Estimator, HasFeaturesCol, HasPredictionCol, HasLabelCol, HasSklearnModel):
    @keyword_only
    def __init__(self, featuresCol="features", predictionCol="prediction", labelCol="label"):
        super(SklearnEstimator, self).__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)
        
    def _fit(self, dataset):
        local_dataset = dataset.withColumn(self.getFeaturesCol(), _sparseVectorToArray("word_vector")).localCheckpoint()
        local_dataset = local_dataset.select(self.getFeaturesCol(), self.getLabelCol()).toPandas()
        self.est = LogisticRegression()
        self.est.fit(local_dataset[self.getFeaturesCol()].tolist(), local_dataset[self.getLabelCol()])
        self.sklearn_model = base64.b64encode(pickle.dumps(self.est)).decode('utf-8')
        #self.setSklearnModel(base64.b64encode(pickle.dumps(self.est)).decode('utf-8'))
        #print(pickle.loads(base64.b64decode(self.getSklearnModel().encode('utf-8'))))
        return SklearnEstimatorModel(sklearn_model=self.sklearn_model, predictionCol=self.getPredictionCol(),
                                         featuresCol=self.getFeaturesCol(), labelCol=self.getLabelCol())

tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
hasher = HashingTF(numFeatures=10, binary=True, inputCol=tokenizer.getOutputCol(), outputCol="word_vector")

spark_est = SklearnEstimator()

pipeline = Pipeline(stages=[
    tokenizer,
    hasher,
    spark_est
])

