#!/opt/conda/envs/dsenv/bin/python

from pyspark.sql.types import *
from pyspark.ml.feature import *
import pyspark.sql.functions as f
from pyspark.ml import Pipeline

stop_words = StopWordsRemover.loadDefaultStopWords("english")
tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="wordsReview", pattern="\\W")
swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="reviewFiltered", stopWords=stop_words)
count_vectorizer = CountVectorizer(inputCol=swr.getOutputCol(), outputCol="reviewVector", binary=True, vocabSize=2500)
assembler = VectorAssembler(inputCols=[count_vectorizer.getOutputCol(), 'verified'], outputCol="features")

pipeline = Pipeline(stages=[
    tokenizer,   
    swr,    
    count_vectorizer,
    assembler
])
