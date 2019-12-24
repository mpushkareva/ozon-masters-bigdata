#!/opt/conda/envs/dsenv/bin/python
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
    predictions = est.predict(series.tolist())
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
    sklearn_model = Param(Params._dummy(), "sklearn_model", "sklearn_model",
        typeConverter=TypeConverters.toString)
    
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
        #est_broadcast = spark.sparkContext.broadcast(self.est)
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
