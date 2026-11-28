import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("PositionalIndex") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

tokens_data = spark.read.csv(
    "output/tokens.csv", 
    header=True, 
    inferSchema=True
)

# Make sure that the positions is int
tokens_data = tokens_data.withColumn(
    "position",
    F.col("position").cast("int")
)

tokens_data.show(100, truncate=False)

# 1. Build Positional Index
positional_index = tokens_data.groupBy(
    "term",
    "doc_id"
).agg(
    F.sort_array(
        F.collect_list("position")
    ).alias("positions")
)

# 2. Sort the Positional Index
positional_index = positional_index.orderBy("term", "doc_id")

print("FLAT POSITIONAL INDEX")
positional_index.show(100, truncate=False)

# 3. Build complete positional index
complete_positional_index = positional_index.groupBy("term").agg(
    F.sort_array(
        F.collect_list(
            F.struct("doc_id", "positions")
        )
    ).alias("postings")
)

complete_positional_index = complete_positional_index.orderBy("term")

print("COMPLETE POSITIONAL INDEX")
complete_positional_index.show(100, truncate=False)

# 4. Test Repeated Terms
repeated_terms = positional_index.filter(
    F.size("positions") > 1
)

print("REPEATED TERMS TEST")
repeated_terms.show(100, truncate=False)

# 5. Test Specific Word Positions
test_result = positional_index.filter(
    (F.col("term") == "online") &
    (F.col("doc_id") == "d5")
)

print("POSITION VALIDATION TEST")
test_result.show(truncate=False)

# 6. Save Positional Index to CSV

if not os.path.exists("output"):
    os.makedirs("output")

nested_data = complete_positional_index.toPandas()

nested_data["postings"] = nested_data["postings"].apply(
    lambda docs: str([
        {
            "doc_id": doc["doc_id"],
            "positions": doc["positions"]
        }
        for doc in docs
    ])
)

nested_data.to_csv(
    "output/positional_index.csv",
    index=False
)

print("Positional Index Created Successfully!")

spark.stop()