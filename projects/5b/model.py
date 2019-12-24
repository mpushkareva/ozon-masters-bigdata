#!/opt/conda/envs/dsenv/bin/python
from pyspark.sql.types import *
from pyspark.ml.feature import *
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from sklearn_wrapper import SklearnEstimatorModel

    
stop_words = StopWordsRemover.loadDefaultStopWords("english")
tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="wordsReview", pattern="\\W")
swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="reviewFiltered", stopWords=stop_words)
count_vectorizer = CountVectorizer(inputCol=swr.getOutputCol(), outputCol="reviewVector", binary=True, vocabSize=20)
assembler = VectorAssembler(inputCols=[count_vectorizer.getOutputCol(), 'verified'], outputCol="features")

@F.pandas_udf(DoubleType())
def predict(series):
    # Необходимо сделать преобразования, потому что на вход приходит pd.Series(list)
    print(est_broadcast.value)
    predictions = est_broadcast.value.predict(series.tolist())
    return pd.Series(predictions)

@F.udf(ArrayType(DoubleType()))
def vectorToArray(row):
    return row.toArray().tolist()

class HasSklearnModel(Params):

    sklearn_model = Param(Params._dummy(), "sklearn_model", "sklearn_model",
        typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasSklearnModel, self).__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)
        
    def setSklearnModel(self, value):
        return self._set(sklearn_model=value)

    def getSklearnModel(self):
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
        self.est = loads(base64.b64decode(self.getSklearnModel().encode('utf-8')))
        kwargs = self._input_kwargs
        return self._set(**kwargs)
        
    def _transform(self, dataset):
        global est_broadcast 
        est_broadcast = spark.sparkContext.broadcast(self.est)
        dataset = dataset.withColumn("features_array", vectorToArray("features")).localCheckpoint()
        return dataset.withColumn("prediction", predict("features_array"))

class SklearnEstimator(Estimator, HasFeaturesCol, HasPredictionCol, HasLabelCol, HasSklearnModel):
    @keyword_only
    def __init__(self, featuresCol="features", predictionCol="prediction", labelCol="label"):
        super(SklearnEstimator, self).__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)
        
    def _fit(self, dataset):
        local_dataset = dataset.withColumn("features_array", vectorToArray("features")).localCheckpoint()
        local_dataset = dataset.select("features_array", self.getLabelCol()).toPandas()
        self.est = LogisticRegression()
        #with open(self.model_file, "wb") as f:
         #   pickle.dump(self.est, f)    
        self.est.fit(local_dataset["features_array"].tolist(), local_dataset[self.getLabelCol()])
        model_string = base64.b64encode(dumps(self.est)).decode('utf-8')
        self.setSklearnModel(model_string)
        return SklearnEstimatorModel(sklearn_model=model_string, predictionCol=self.getPredictionCol(),
                                         featuresCol="features_array")



spark_est = SklearnEstimator()

pipeline = Pipeline(stages=[
    tokenizer,   
    swr,    
    count_vectorizer,
    assembler,
    spark_est
])

