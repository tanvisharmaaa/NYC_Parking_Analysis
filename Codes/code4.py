#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize SparkSession
spark = SparkSession.builder.appName("RevenuePredictionWithRegularization").getOrCreate()

# Paths for input and output
input_path = "gs://finalprojc/revenue_per_precinct.csv"
model_output_path = "gs://finalprojc/revenue_prediction_model_regularized"
predictions_output_path = "gs://finalprojc/revenue_predictions_regularized"

# Load manipulated data
data = spark.read.csv(input_path, header=True, inferSchema=True)

# Ensure no missing values
data = data.na.drop()

# Assemble features into a single vector
assembler = VectorAssembler(
    inputCols=["ViolationCount"], 
    outputCol="features"
)
data = assembler.transform(data)

# Split data into training and testing sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Initialize Linear Regression with regularization
lr = LinearRegression(
    featuresCol="features", 
    labelCol="EstimatedRevenue", 
    maxIter=10, 
    regParam=0.1,  # Add L2 regularization (default: 0.0)
    elasticNetParam=0.0  # Pure L2 regularization
)

# Train the model
model = lr.fit(train_data)
print("Model training successful with regularization!")

# Save the trained model
model.write().overwrite().save(model_output_path)
print(f"Regularized model saved to: {model_output_path}")

# Make predictions on test data
predictions = model.transform(test_data)

# Evaluate the model
evaluator_rmse = RegressionEvaluator(
    labelCol="EstimatedRevenue", 
    predictionCol="prediction", 
    metricName="rmse"
)
rmse = evaluator_rmse.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

evaluator_r2 = RegressionEvaluator(
    labelCol="EstimatedRevenue", 
    predictionCol="prediction", 
    metricName="r2"
)
r2 = evaluator_r2.evaluate(predictions)
print(f"R² (Coefficient of Determination): {r2:.2f}")

# Save predictions to GCS
predictions.select("ViolationCount", "EstimatedRevenue", "prediction").write.csv(
    predictions_output_path, mode="overwrite", header=True
)
print(f"Predictions saved to: {predictions_output_path}")



