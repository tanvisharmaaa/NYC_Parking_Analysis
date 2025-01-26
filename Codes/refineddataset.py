#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, dayofmonth, hour, col, lit, count  # Fixed imports
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel

# Initialize SparkSession
spark = SparkSession.builder.appName("ViolationPrediction").getOrCreate()

# Paths for data, model, and outputs
input_path = "gs://term-project-cs777/merged_df.csv"
model_path = "gs://term-project-cs777/violation-prediction-model"
output_revenue_path = "gs://term-project-cs777/revenue_per_precinct.csv"

# Load merged dataset
data = spark.read.csv(input_path, header=True, inferSchema=True)

# Check available columns
print("Available columns:", data.columns)

# Define default values for existing columns only
default_values = {
    "Summons Number": 0,
    "Vehicle Make": "unknown",
    "Plate ID": -1,
    "Violation Precinct": "unknown",
    "Issuer Precinct": "unknown",
    "Issuing Agency": "unknown",
    "Violation Code": -1,
    "Violation Time": "00:00",
    "Feet From Curb": -1,
    "Street Name": "unknown",
    "Plate Type": "unknown",
    "Vehicle Body Type": "unknown",
    "Registration State": "unknown",
}

# Filter default_values to include only existing columns
filtered_defaults = {key: value for key, value in default_values.items() if key in data.columns}

# Fill null values with filtered defaults
data = data.fillna(filtered_defaults)

# Revenue evaluation: Estimated revenue per precinct
fine_amount = 50  # Assumed fine per violation
revenue_per_precinct = data.groupBy("Violation Precinct").agg(
    count("*").alias("ViolationCount"),  # Fixed missing count function
    (count("*") * lit(fine_amount)).alias("EstimatedRevenue")
)

# Show revenue per precinct
print("Revenue per precinct:")
revenue_per_precinct.show()

# Save revenue to GCS
try:
    revenue_per_precinct.write.csv(output_revenue_path, mode="overwrite", header=True)
    print(f"Revenue results saved to: {output_revenue_path}")
except Exception as e:
    print(f"Failed to save revenue results: {e}")

# Feature Engineering: Extract date-time and vehicle-related features
data = data.withColumn("Year", year(col("Issue Date"))) \
           .withColumn("Month", month(col("Issue Date"))) \
           .withColumn("Day", dayofmonth(col("Issue Date"))) \
           .withColumn("Hour", hour(col("Issue Date")))

# Select required columns for prediction
data = data.select("Year", "Month", "Day", "Hour", "Vehicle Body Type", "Violation Code")

# Handle categorical data
vehicle_indexer = StringIndexer(inputCol="Vehicle Body Type", outputCol="VehicleBodyTypeIndexed", handleInvalid="keep")
assembler = VectorAssembler(
    inputCols=["Year", "Month", "Day", "Hour", "VehicleBodyTypeIndexed"],
    outputCol="features",
    handleInvalid="skip"
)

# Logistic Regression model
lr = LogisticRegression(featuresCol="features", labelCol="Violation Code", maxIter=10)

# Pipeline
pipeline = Pipeline(stages=[vehicle_indexer, assembler, lr])

# Train-test split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Train the model and handle potential failures
try:
    model = pipeline.fit(train_data)
    print("Model training successful.")

    # Save the trained model to GCS
    model.write().overwrite().save(model_path)
    print(f"Model saved to: {model_path}")

    # Evaluate the model
    predictions = model.transform(test_data)
    predictions.select("features", "Violation Code", "prediction").show(10)

    # Use the trained model for new predictions
    loaded_model = PipelineModel.load(model_path)
    new_predictions = loaded_model.transform(test_data)
    new_predictions.select("features", "Violation Code", "prediction").show(10)

except Exception as e:
    print(f"Model training or prediction failed: {e}")


