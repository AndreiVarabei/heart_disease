from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("heart_disease").getOrCreate()

df = spark.read.csv(path="data_cardiovascular_risk.csv", inferSchema=True, header=True)
