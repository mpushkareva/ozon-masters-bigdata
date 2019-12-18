from pyspark.sql.types import *
from pyspark.ml.feature import *
import pyspark.sql.functions as f
#from pyspark.ml import Estimator
from sklearn.linear_model import LogisticRegression
import pandas as pd
from joblib import load

class SklearnEstimatorModel(Model, HasFeaturesCol, HasLabelCol, HasPredictionCol):
    model_file = Param(Params._dummy(), "model_file",
                      "path to pickled scikit-learn logistic regression model",
                      typeConverter=TypeConverters.toString)
    @keyword_only
    def __init__(self, model_file=None, featuresCol="features", labelCol="label", predictionCol="prediction"):
        super(SKLogisticRegreesionModel, self).__init__()
        if model_file is None:
            raise ValueError("model_file must be specified!")
        with open(model_file, "rb") as f:
            self.estimator = load.load(model_file)
        kwargs = self._input_kwargs
        self._set(**kwargs)
        
    def _transform(self, dataset):
        dataset = dataset.withColumn("features_array", vectorToArray(self.getFeaturesCol())).localCheckpoint()
        return dataset.withColumn(self.getPredictionCol(), predict("features_array"))