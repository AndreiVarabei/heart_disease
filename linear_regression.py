from pyspark.sql import SparkSession

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder.appName("heart_disease").getOrCreate()

df = spark.read.csv(path="data_cardiovascular_risk.csv", inferSchema=True, header=True)

df.printSchema()

#count rows before drop null rows
print(df.count())

droped = df.dropna()

#count rows after drop null rows
print(droped .count())

#replace categorical columns with numeric columns
indexer = StringIndexer(inputCols=["sex", "is_smoking"], outputCols=["sex_cat","is_smoking_cat"])
indexed = indexer.fit(droped).transform(droped)

#create features Vector
assembler = VectorAssembler(inputCols=['age', 'education', 'cigsPerDay', 'BPMeds', 'prevalentStroke',
                             'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI',
                             'heartRate', 'glucose', 'sex_cat', 'is_smoking_cat'],
                            outputCol="features")
output = assembler.transform(indexed)
output.select("features", "TenYearCHD").show()



