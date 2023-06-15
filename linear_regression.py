from pyspark.sql import SparkSession
from pyspark.sql.functions import corr

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.appName("heart_disease").getOrCreate()

df = spark.read.csv(path="data_cardiovascular_risk.csv", inferSchema=True, header=True)

df.printSchema()

#count rows before drop null rows
print(df.count())

droped = df.dropna()

#count rows after drop null rows
print(droped.count())

#replace categorical columns with numeric columns
indexer = StringIndexer(inputCols=["sex", "is_smoking"], outputCols=["sex_cat","is_smoking_cat"])
indexed = indexer.fit(droped).transform(droped)

#create features Vector
assembler = VectorAssembler(inputCols=['age', 'education', 'cigsPerDay', 'BPMeds', 'prevalentStroke',
                             'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI',
                             'heartRate', 'glucose', 'sex_cat', 'is_smoking_cat'],
                            outputCol="features")
output = assembler.transform(indexed)
final_data = output.select(["features", "TenYearCHD"])

#divide the data to train and test data and check distribution
train_data, test_data = final_data.randomSplit([0.7, 0.3])

train_data.describe().show()

test_data.describe().show()

# train data and evaluete test data
disease_lr = LinearRegression(labelCol="TenYearCHD")

trained_disease_model = disease_lr.fit(train_data)

prediction_results = trained_disease_model.evaluate(test_data)

#check results

prediction_results.residuals.show(50)

print(prediction_results.r2)

print(prediction_results.rootMeanSquaredError)

# The most correlation column with TenYearCHD
indexed.select(corr(col1='TenYearCHD', col2='age')).show()

