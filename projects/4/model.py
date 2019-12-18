#!/opt/conda/envs/dsenv/bin/python

from pyspark.sql.types import *
from pyspark.ml.feature import *
import pyspark.sql.functions as f
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

stop_words = StopWordsRemover.loadDefaultStopWords("english")
tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="wordsReview", pattern="\\W")
swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="reviewFiltered", stopWords=stop_words)
count_vectorizer = CountVectorizer(inputCol=swr.getOutputCol(), outputCol="reviewVector", binary=True, vocabSize=2500)
assembler = VectorAssembler(inputCols=[count_vectorizer.getOutputCol(), 'verified'], outputCol="features")

lr = LogisticRegression(featuresCol="reviewVector", labelCol="overall", maxIter=5, regParam=0)

evaluator = RegressionEvaluator(labelCol="overall", predictionCol="prediction", metricName='rmse')

pipeline = Pipeline(stages=[
    tokenizer,   
    swr,    
    count_vectorizer,
    assembler,
    lr
])
