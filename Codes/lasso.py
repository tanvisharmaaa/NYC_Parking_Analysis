#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Initialize Spark session
spark = SparkSession.builder.appName("RevenueForecastLasso").getOrCreate()

# Load the dataset
input_path = "gs://term-project-cs777/refined_df.csv"
data = spark.read.csv(input_path, header=True, inferSchema=True)

# Add an index column to simulate a time series
data = data.withColumn("Index", monotonically_increasing_id())

# Assemble features for regression
assembler = VectorAssembler(inputCols=["Index"], outputCol="features")
data = assembler.transform(data)

# Split into training and testing sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Train a LASSO Regression model (Linear Regression with L1 regularization)
lasso = LinearRegression(featuresCol="features", labelCol="EstimatedRevenue", regParam=0.1, elasticNetParam=1.0)
lasso_model = lasso.fit(train_data)

# Save the trained model
model_path = "gs://term-project-cs777/revenue_forecast_lasso_model"
lasso_model.write().overwrite().save(model_path)
print(f"LASSO Model saved to: {model_path}")

# Evaluate the model
test_results = lasso_model.evaluate(test_data)
print(f"Root Mean Squared Error (RMSE): {test_results.rootMeanSquaredError}")
print(f"R² (Coefficient of Determination): {test_results.r2}")

# Generate predictions for test data
predictions = lasso_model.transform(test_data)

# Save predictions to GCS
predictions_path = "gs://term-project-cs777/revenue_forecast_lasso_predictions"
predictions.select("Index", "EstimatedRevenue", "prediction").write.csv(predictions_path, mode="overwrite", header=True)
print(f"LASSO Predictions saved to: {predictions_path}")

