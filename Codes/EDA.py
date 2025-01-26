#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, when, count
import matplotlib.pyplot as plt

# ============================
# 1. Initialize Spark Session
# ============================
spark = SparkSession.builder \
    .appName("NYC Parking Violations Analysis") \
    .getOrCreate()

# Load the dataset
file_path = "gs://term-project-cs777/merged_df.csv"  # Replace with your dataset path
data = spark.read.csv(file_path, header=True, inferSchema=True)

# ============================
# 2. Basic EDA and Missing Values
# ============================
# Count total rows and columns
total_rows = data.count()
total_columns = len(data.columns)
print(f"Total Rows: {total_rows}, Total Columns: {total_columns}")

# Count missing values for each column
missing_values = data.select([(total_rows - count(c)).alias(c) for c in data.columns])
missing_values.show()

# ============================
# 3. Top Violations and Precincts
# ============================
# Top 10 Violation Precincts
top_violations = data.groupBy("Violation Precinct").count().orderBy(col("count").desc()).limit(10)
print("\nTop 10 Violation Precincts:")
top_violations.show()

# Top 10 Issuing Precincts
top_precincts = data.groupBy("Issuer Precinct").count().orderBy(col("count").desc()).limit(10)
print("\nTop 10 Issuing Precincts:")
top_precincts.show()

# ============================
# 4. Parsing "Violation Time"
# ============================
# Extract valid times from "Violation Time" (HHMMP format)
data = data.withColumn("Violation Time Valid", regexp_extract(col("Violation Time"), r"^(\d{4}[AP])$", 0))

# Extract hour using custom logic
data = data.withColumn(
    "Violation Hour",
    when(col("Violation Time Valid").rlike(r"^[0-9]{4}[AP]$"), 
         when(col("Violation Time Valid").substr(5, 1) == "P", 
              (col("Violation Time Valid").substr(1, 2).cast("int") % 12) + 12
         ).otherwise(
              col("Violation Time Valid").substr(1, 2).cast("int") % 12
         )
    ).otherwise(None)
)

# ============================
# 5. Top Vehicle Types
# ============================
top_vehicle_types = data.groupBy("Vehicle Body Type").count().orderBy(col("count").desc()).limit(10)
print("\nTop 10 Vehicle Types with Violations:")
top_vehicle_types.show()

# ============================
# 6. Visualizations (Matplotlib)
# ============================
# Convert necessary data to Pandas for plotting
violation_hours = data.groupBy("Violation Hour").count().orderBy(col("Violation Hour").asc()).toPandas()

# Plot violations by hour
plt.figure(figsize=(12, 6))
plt.bar(violation_hours["Violation Hour"], violation_hours["count"], color="skyblue")
plt.title("Violations by Hour of the Day")
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Violations")
plt.xticks(rotation=45)
plt.show()

# ============================
# 7. Save Processed Data
# ============================
# Save the processed dataset back to CSV
output_path = "processed_nyc_violations.csv"  # Replace with your desired path
data.write.csv(output_path, header=True, mode="overwrite")
print(f"\nProcessed data saved to: {output_path}")

