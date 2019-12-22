from pyspark.sql.types import *
from pyspark.ml.feature import *
import pyspark.sql.functions as F
#from pyspark.ml import Estimator
from sklearn.linear_model import LogisticRegression
import pandas as pds
from pyspark import keyword_only
from pyspark.ml import Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol, HasPredictionCol
from joblib import load

@F.udf(ArrayType(DoubleType()))
def vectorToArray(row):
    return row.toArray().tolist()

est_broadcast = spark.sparkContext.broadcast(est)
@F.pandas_udf(DoubleType())
def predict(series):
    predictions = est_broadcast.value.predict(series.tolist())
    return pd.Series(predictions)

class SklearnEstimatorModel(Model, HasFeaturesCol, HasLabelCol, HasPredictionCol):
    model_file = Param(Params._dummy(), "model_file",
                      "path to pickled scikit-learn logistic regression model",
                      typeConverter=TypeConverters.toString)
    @keyword_only
    def __init__(self, model_file=None, featuresCol="features", labelCol="label", predictionCol="prediction"):
        super(SklearnEstimatorModel, self).__init__()
        if model_file is None:
            raise ValueError("model_file must be specified!")
        with open(model_file, "rb") as f:
            self.estimator = load(model_file)
        kwargs = self._input_kwargs
        self._set(**kwargs)
        
    def _transform(self, dataset):
        dataset = dataset.withColumn("features_array", vectorToArray(self.getFeaturesCol())).localCheckpoint()
        return dataset.withColumn(self.getPredictionCol(), predict("features_array"))
