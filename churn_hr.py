# Databricks notebook source
# MAGIC %sql 
# MAGIC SELECT * 
# MAGIC FROM telco_churn.employee_attrition

# COMMAND ----------

churn = spark.sql('''
SELECT *
FROM telco_churn.employee_attrition
''')
cols = churn.columns

# COMMAND ----------

churn.drop("Over18") \
  .printSchema()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.types import * 
import pyspark.sql.functions as F
from pyspark.sql.functions import col, asc,desc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyspark.sql import SQLContext
from pyspark.mllib.stat import Statistics
import pandas as pd
from pyspark.sql.functions import udf
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler,StandardScaler
from pyspark.ml import Pipeline
from sklearn.metrics import confusion_matrix


# COMMAND ----------

churn.printSchema()

# COMMAND ----------

def numeric_describe(df):
  numeric_features = [t[0] for t in churn.dtypes if t[1] == 'int']
  final = churn.select(numeric_features).describe().toPandas().transpose()
  return final

# COMMAND ----------

def null_check(df):
  '''
    Check for nulls in a spark dataframe
    
    Args:
        df (object): dataset rendered into a spark dataframe
        
    Returns:
        Counts of nulls within a spark dataframe
        
        Example:
            null_check(test_normal_idx, data_normal_dir, test_normal, imgs_normal)
  '''
  from pyspark.sql.functions import isnan, when, count, col
  null = df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).toPandas().head()
  return null

# COMMAND ----------

def hist(df):
  '''
  Returns 
  '''
  from matplotlib import cm
  fig = plt.figure(figsize=(25,15)) ## Plot Size 
  st = fig.suptitle("Distribution of Features", fontsize=50,
                    verticalalignment='center') # Plot Main Title 

  for col,num in zip(df.toPandas().describe().columns, range(1,11)):
      ax = fig.add_subplot(3,4,num)
      ax.hist(df.toPandas()[col])
      plt.style.use('dark_background') 
      plt.grid(False)
      plt.xticks(rotation=45,fontsize=20)
      plt.yticks(fontsize=15)
      plt.title(col.upper(),fontsize=20)
  plt.tight_layout()
  st.set_y(0.95)
  fig.subplots_adjust(top=0.85,hspace = 0.4)
  plt.show()

# COMMAND ----------

def summary_stats(df):
  '''
   Returns statistics of the data frame: count, mean, std deviation, min, and max values for each column 
    
    Args:
        df (object): dataset rendered into a spark dataframe
        
    Returns:
        Counts of nulls within a spark dataframe
        
        Example:
            null_check(test_normal_idx, data_normal_dir, test_normal, imgs_normal)
  '''
  numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']
  df.select(numeric_features).describe().toPandas().transpose()

# COMMAND ----------

# DBTITLE 1,Statistics
numeric_features = [t[0] for t in churn.dtypes if t[1] == 'int']
churn.select(numeric_features).describe().toPandas().transpose()

# COMMAND ----------

# DBTITLE 1,Attrition Counts
churn.groupby("attrition").count().show()

# COMMAND ----------

# DBTITLE 1,Check for Nulls
null_check(churn)

# COMMAND ----------

churn.toPandas()

# COMMAND ----------

categorical_columns = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus','OverTime']
numeric_cols = ['Age', 'DistanceFromHome', 'Education', 'EmployeeCount', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 
                'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', "TotalWorkingYears", 
                'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

# COMMAND ----------

import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler

from distutils.version import LooseVersion

categoricalColumns = categorical_columns
stages = [] # stages in Pipeline
for categoricalCol in categoricalColumns:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    if LooseVersion(pyspark.__version__) < LooseVersion("3.0"):
        from pyspark.ml.feature import OneHotEncoderEstimator
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    else:
        from pyspark.ml.feature import OneHotEncoder
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]

# COMMAND ----------

# Convert label into label indices using the StringIndexer
label_stringIdx = StringIndexer(inputCol= "Gender", outputCol="label")
stages += [label_stringIdx]

# COMMAND ----------

# Transform all features into a vector using VectorAssembler
numericCols = numeric_cols
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
  
partialPipeline = Pipeline().setStages(stages)
pipelineModel = partialPipeline.fit(churn)
preppedDataDF = pipelineModel.transform(churn)

# COMMAND ----------

# Fit model to prepped data
lrModel = LogisticRegression().fit(preppedDataDF)

# ROC for training data
display(lrModel, preppedDataDF, "ROC")

# COMMAND ----------

display(lrModel, preppedDataDF)

# COMMAND ----------

# Keep relevant columns
selectedcols = ["label", "features"] + cols
dataset = preppedDataDF.select(selectedcols)
display(dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocess Data

# COMMAND ----------

### Randomly split data into training and test sets. set seed for reproducibility
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
print(trainingData.count())
print(testData.count())

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# Create initial LogisticRegression model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

# Train model with Training Data
lrModel = lr.fit(trainingData)

# COMMAND ----------

# Make predictions on test data using the transform() method.
# LogisticRegression.transform() will only use the 'features' column.
predictions = lrModel.transform(testData)

# COMMAND ----------

# View model's predictions and probabilities of each prediction class
# You can select any columns in the above schema to view as well
selected = predictions.select("label", "prediction", "probability", "age", "overtime")
display(selected)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Evaluate model
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)

# COMMAND ----------

evaluator.getMetricName()

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) #lasso, ridge, and elastic net 
             .addGrid(lr.maxIter, [1, 5, 10])
             .build())

# COMMAND ----------

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Run cross validations
cvModel = cv.fit(trainingData)
# this will likely take a fair amount of time because of the amount of models that we're creating and testing

# COMMAND ----------

predictions = cvModel.transform(testData)

# COMMAND ----------

# cvModel uses the best model found from the Cross Validation
# Evaluate best model
evaluator.evaluate(predictions)

# COMMAND ----------

print('Model Intercept: ', cvModel.bestModel.intercept)

# COMMAND ----------

weights = cvModel.bestModel.coefficients
weights = [(float(w),) for w in weights]  # convert numpy type to float, and to tuple
weightsDF = sqlContext.createDataFrame(weights, ["Feature Weight"])
display(weightsDF)

# COMMAND ----------

# View best model's predictions and probabilities of each prediction class
selected = predictions.select("label", "prediction", "probability", "age", "overtime")
display(selected)
