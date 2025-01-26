#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor, GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize SparkSession
spark = SparkSession.builder.appName("RevenueForecasting").getOrCreate()

# Input dataset path
input_path = "gs://term-project-cs777/refined_df.csv"

# Read data
data = spark.read.csv(input_path, header=True, inferSchema=True)

# Feature engineering
assembler = VectorAssembler(inputCols=["ViolationCount"], outputCol="features")
data = assembler.transform(data)

# Split data into train and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Define a single evaluator for all models
evaluator_rmse = RegressionEvaluator(labelCol="EstimatedRevenue", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="EstimatedRevenue", predictionCol="prediction", metricName="r2")

# --- Train and Evaluate Decision Tree Regressor ---
print("Training Decision Tree Regressor...")
dt = DecisionTreeRegressor(featuresCol="features", labelCol="EstimatedRevenue")
dt_model = dt.fit(train_data)

# Save Decision Tree model
dt_model_path = "gs://term-project-cs777/revenue_forecast_decision_tree_model"
dt_model.write().overwrite().save(dt_model_path)
print(f"Decision Tree Model saved to: {dt_model_path}")

# Evaluate Decision Tree model
dt_predictions = dt_model.transform(test_data)
dt_rmse = evaluator_rmse.evaluate(dt_predictions)
dt_r2 = evaluator_r2.evaluate(dt_predictions)

# Save predictions
dt_predictions_path = "gs://term-project-cs777/revenue_forecast_decision_tree_predictions"
dt_predictions.select("Violation Precinct", "EstimatedRevenue", "prediction").write.csv(dt_predictions_path, mode="overwrite", header=True)
print(f"Decision Tree Predictions saved to: {dt_predictions_path}")
print(f"Decision Tree RMSE: {dt_rmse}")
print(f"Decision Tree R²: {dt_r2}")

# --- Train and Evaluate Gradient Boosted Trees Regressor ---
print("\nTraining Gradient Boosted Trees Regressor...")
gbt = GBTRegressor(featuresCol="features", labelCol="EstimatedRevenue", maxIter=10)
gbt_model = gbt.fit(train_data)

# Save GBT model
gbt_model_path = "gs://term-project-cs777/revenue_forecast_gbt_model"
gbt_model.write().overwrite().save(gbt_model_path)
print(f"GBT Model saved to: {gbt_model_path}")

# Evaluate GBT model
gbt_predictions = gbt_model.transform(test_data)
gbt_rmse = evaluator_rmse.evaluate(gbt_predictions)
gbt_r2 = evaluator_r2.evaluate(gbt_predictions)

# Save predictions
gbt_predictions_path = "gs://term-project-cs777/revenue_forecast_gbt_predictions"
gbt_predictions.select("Violation Precinct", "EstimatedRevenue", "prediction").write.csv(gbt_predictions_path, mode="overwrite", header=True)
print(f"GBT Predictions saved to: {gbt_predictions_path}")
print(f"GBT RMSE: {gbt_rmse}")
print(f"GBT R²: {gbt_r2}")

# --- Train and Evaluate Random Forest Regressor ---
print("\nTraining Random Forest Regressor...")
rf = RandomForestRegressor(featuresCol="features", labelCol="EstimatedRevenue", numTrees=20)
rf_model = rf.fit(train_data)

# Save Random Forest model
rf_model_path = "gs://term-project-cs777/revenue_forecast_rf_model"
rf_model.write().overwrite().save(rf_model_path)
print(f"Random Forest Model saved to: {rf_model_path}")

# Evaluate Random Forest model
rf_predictions = rf_model.transform(test_data)
rf_rmse = evaluator_rmse.evaluate(rf_predictions)
rf_r2 = evaluator_r2.evaluate(rf_predictions)

# Save predictions
rf_predictions_path = "gs://term-project-cs777/revenue_forecast_rf_predictions"
rf_predictions.select("Violation Precinct", "EstimatedRevenue", "prediction").write.csv(rf_predictions_path, mode="overwrite", header=True)
print(f"Random Forest Predictions saved to: {rf_predictions_path}")
print(f"Random Forest RMSE: {rf_rmse}")
print(f"Random Forest R²: {rf_r2}")

print("\nTraining and evaluation completed for all models!")

