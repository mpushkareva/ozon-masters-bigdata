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

pipeline = Pipeline(stages=[
    tokenizer,   
    swr,    
    count_vectorizer,
    assembler
])

@F.udf(ArrayType(DoubleType()))
def vectorToArray(row):
    return row.toArray().tolist()

@F.pandas_udf(DoubleType())
def predict(series):
    predictions = est_broadcast.value.predict(series.tolist())
    return pd.Series(predictions)

sklearn_est = SklearnEstimatorModel

