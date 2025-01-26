#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.system("pip install plotly -q")
os.system("pip install kaleido -q")
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from google.cloud import storage
import matplotlib.pyplot as plt
import plotly.graph_objs as go


# Initialize SparkSession
spark = SparkSession.builder \
    .appName("NYC Parking Violations on GCP") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Define GCP bucket information
BUCKET_NAME = "term-project-cs777"  # Replace with your bucket name
GRAPH_FOLDER = ""  # Empty for root directory

# Define a function to upload graphs to GCP
def upload_to_gcp(bucket_name, source_file_name, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to gs://{bucket_name}/{destination_blob_name}")

# Read data from GCP bucket
data_path = "gs://term-project-cs777/merged_df.csv"  # Replace with your dataset path
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Data Cleaning
df = df.withColumn("Issue Date", to_date("Issue Date", "MM/dd/yyyy")) \
       .withColumn("Vehicle Expiration Date", to_date("Vehicle Expiration Date", "MM/dd/yyyy")) \
       .withColumn('Registration State', when(col('Registration State') == '99', 'unknown').otherwise(col('Registration State'))) \
       .withColumn('Plate Type', when(col('Plate Type') == '999', 'unknown').otherwise(col('Plate Type')))

# Filter out columns with more than 85% null values
null_percentages = df.select([
    (count(when(col(c).isNull(), c)) / count("*")).alias(c) for c in df.columns
])

null_cols = [
    c for c in null_percentages.columns
    if null_percentages.select(c).first()[0] is not None and null_percentages.select(c).first()[0] > 0.85
]

df = df.drop(*null_cols)
print(f"Columns with more than 85% null values: {null_cols}")

# Fill null values with defaults
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
    "Violation Description": "unknown",
    "Vehicle Body Type": "unknown",
    "Registration State": "unknown"
}
df = df.fillna(default_values)

# Hypothesis 1: Top 10 Streets with Most Fines
street_fines = df.groupBy("Street Name").agg(count("Summons Number").alias("Num Fines")).orderBy(desc("Num Fines"))
top10_streets = street_fines.limit(10).toPandas()

plt.figure(figsize=(12, 6))
plt.bar(top10_streets["Street Name"], top10_streets["Num Fines"], color='pink')
plt.xticks(rotation=90)
plt.xlabel("Street Name")
plt.ylabel("Number of Fines")
plt.title("Top 10 Streets with Most Fines")
local_path = "top10_streets.png"
plt.savefig(local_path)
upload_to_gcp(BUCKET_NAME, local_path, local_path)
os.remove(local_path)

# Hypothesis 2: Luxury Cars Ticket Count
luxury_cars = ['Mercedes-Benz', 'BMW', 'Audi', 'Lexus', 'Porsche']
violation_count = df.groupBy("Vehicle Make").agg(count("Violation Post Code").alias("vio_count"))
violation_count = violation_count.withColumn("vio_count_int", col("vio_count").cast("integer"))
luxury_violations = violation_count.filter(col("Vehicle Make").isin(luxury_cars)).selectExpr("sum(vio_count_int)").collect()[0][0]
total_violations = violation_count.selectExpr("sum(vio_count_int)").collect()[0][0]
luxury_fraction = luxury_violations / total_violations
print(f"Luxury Cars Fraction of Violations: {luxury_fraction:.2%}")

# Hypothesis 3: Top 10 Violations
violation_counts = df.groupBy("Violation Description").agg(count("Summons Number").alias("ticket_count"))
top10_violations = violation_counts.orderBy(desc("ticket_count")).limit(10).toPandas()

fig = go.Figure()
fig.add_trace(go.Bar(x=top10_violations["Violation Description"], y=top10_violations["ticket_count"], marker_color='indianred'))
fig.update_layout(title_text="Top 10 Violations", xaxis_title="Violation Description", yaxis_title="Ticket Count")
local_path = "top10_violations.png"
fig.write_image(local_path)
upload_to_gcp(BUCKET_NAME, local_path, local_path)
os.remove(local_path)

# Hypothesis 4: Top Issuing Agencies
agency_counts = df.groupBy("Issuing Agency").agg(count("Summons Number").alias("ticket_count")).orderBy(desc("ticket_count"))
top_agencies = agency_counts.limit(5).toPandas()

fig = go.Figure()
for i, agency in enumerate(top_agencies.iterrows()):
    fig.add_trace(go.Bar(x=[agency[1]["Issuing Agency"]], y=[agency[1]["ticket_count"]], name=agency[1]["Issuing Agency"]))
fig.update_layout(title="Top Issuing Agencies", xaxis_title="Agency", yaxis_title="Ticket Count")
local_path = "top_agencies.png"
fig.write_image(local_path)
upload_to_gcp(BUCKET_NAME, local_path, local_path)
os.remove(local_path)

