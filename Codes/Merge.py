#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import random

# Initialize Spark session
spark = SparkSession.builder.appName("MergeAndSample").getOrCreate()

# Load the two datasets
file1_path = "gs://term-project-cs777/pt_2016.csv"  # Replace with your file path
file2_path = "gs://term-project-cs777/pt_2017.csv"  # Replace with your file path

df1 = spark.read.csv(file1_path, header=True, inferSchema=True)
df2 = spark.read.csv(file2_path, header=True, inferSchema=True)

# Find common columns
common_columns = set(df1.columns).intersection(set(df2.columns))

# Select only common columns from both dataframes
df1_common = df1.select(*common_columns)
df2_common = df2.select(*common_columns)

# Merge the two datasets
merged_df = df1_common.unionByName(df2_common)

# Save the merged dataset as a single file
merged_df.coalesce(1).write.csv("gs://term-project-cs777/merged_dataset.csv", header=True, mode="overwrite")

# Sample 1% of the data for the test dataset
test_dataset = merged_df.sample(withReplacement=False, fraction=0.01, seed=random.randint(0, 100))

# Save the test dataset as a single file
test_dataset.coalesce(1).write.csv("gs://term-project-cs777/test_dataset.csv", header=True, mode="overwrite")

# Stop the Spark session
spark.stop()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




