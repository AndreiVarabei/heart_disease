from pyspark import Row

from pyspark.sql import SparkSession
from pyspark.sql.functions import corr

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier,\
                                      RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

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
disease_ls = LogisticRegression(featuresCol='features', labelCol="TenYearCHD")
disease_dt = DecisionTreeClassifier(featuresCol='features', labelCol="TenYearCHD")
disease_rf = RandomForestClassifier(featuresCol='features', labelCol="TenYearCHD", numTrees=150)
disease_gb = GBTClassifier(featuresCol='features', labelCol="TenYearCHD")

trained_disease_model_ls = disease_ls.fit(train_data)
trained_disease_model_dt = disease_dt.fit(train_data)
trained_disease_model_rf = disease_rf.fit(train_data)
trained_disease_model_gb = disease_gb.fit(train_data)

prediction_results_ls = trained_disease_model_ls.transform(test_data)
prediction_results_dt = trained_disease_model_dt.transform(test_data)
prediction_results_rf = trained_disease_model_rf.transform(test_data)
prediction_results_gb = trained_disease_model_gb.transform(test_data)

#Check accuracy (tp+tn)/total
eval_accuracy = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='TenYearCHD',
                                                  metricName="accuracy")

print('Logistic regression:', eval_accuracy.evaluate(prediction_results_ls))

print('Decision tree:', eval_accuracy.evaluate(prediction_results_dt))

print('Random forest:', eval_accuracy.evaluate(prediction_results_rf))

print('Gradient boosting:', eval_accuracy.evaluate(prediction_results_gb))


# The most correlation column with TenYearCHD
indexed.select(corr(col1='TenYearCHD', col2='age')).show()

# test with real data
real_data = [Row(age=29, education=4.0,cigsPerDay=0.0, BPMeds=0.0, prevalentStroke=0,
                prevalentHyp=1, diabetes=0, totChol=250.0, sysBP=130.0, diaBP=80.0,
                BMI=26.2, heartRate=72.0, glucose=87, sex_cat=1.0, is_smoking_cat=0.0),
             Row(age=50, education=4.0,cigsPerDay=10.0, BPMeds=1.0, prevalentStroke=1,
                prevalentHyp=1, diabetes=1, totChol=300.0, sysBP=146.0, diaBP=90.0,
                BMI=37.0, heartRate=100.0, glucose=190, sex_cat=1.0, is_smoking_cat=1.0),
             ]
real_df = spark.createDataFrame(real_data)
transform_real_data = assembler.transform(real_df)
real_data = transform_real_data.select(["features"])

# Logistic regression:
real_prediction_ls = trained_disease_model_ls.transform(real_data)
real_prediction_ls.show()

#Decision tree:
real_prediction_dt = trained_disease_model_dt.transform(real_data)
real_prediction_dt.show()

#Random forest:
real_prediction_rf = trained_disease_model_rf.transform(real_data)
real_prediction_rf.show()

# Gradient boosting:
real_prediction_gb = trained_disease_model_gb.transform(real_data)
real_prediction_gb.show()

spark.stop()
