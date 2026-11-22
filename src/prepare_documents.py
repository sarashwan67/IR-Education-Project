import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd


# 1. Create Spark Session
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("IR_Education_Prepare") \
    .getOrCreate()

spark_context = spark.sparkContext
spark_context.setLogLevel("WARN")


# 2. Read Documents
documents = spark_context.wholeTextFiles("data/d*.txt")


# 3. Tokenization
def tokenize(document):

    file_path = document[0]
    content = document[1]

    document_id = os.path.basename(file_path).replace(".txt", "")

    content = content.lower()
    word_list = content.split()

    token_list = []
    word_position = 1

    for word in word_list:
        token_list.append((document_id, word_position, word))
        word_position += 1

    return token_list


# Apply Tokenization
all_tokens = documents.flatMap(tokenize)


# 4. Convert to DataFrame
tokens_data = spark.createDataFrame(
    all_tokens,
    ["doc_id", "position", "term"]
)


# 5. Sort Documents
tokens_data = tokens_data.withColumn(
    "document_number",
    F.regexp_extract("doc_id", r"d(\d+)", 1).cast("int")
)

tokens_data = tokens_data.orderBy("document_number", "position")
tokens_data = tokens_data.drop("document_number")

tokens_data.show(100, truncate=False)


# 6. Validate Number of Tokens
expected_counts = {
    "d1": 2,
    "d2": 5,
    "d3": 5,
    "d4": 8,
    "d5": 9,
    "d6": 6,
    "d7": 8,
    "d8": 10,
    "d9": 10,
    "d10": 10
}

document_counts = tokens_data.groupBy("doc_id").count().collect()

actual_counts = {}

for row in document_counts:
    actual_counts[row["doc_id"]] = row["count"]


checks_passed = True

for document_id in expected_counts:

    if actual_counts.get(document_id) != expected_counts[document_id]:
        checks_passed = False
        print("Error in:", document_id)
        print("Expected:", expected_counts[document_id])
        print("Actual:", actual_counts.get(document_id))


# 7. Check Word Positions
position_checks = {
    ("online", "d5"): [1, 5, 8],
    ("learning", "d5"): [2, 9],
    ("assessment", "d8"): [2, 4],
    ("student", "d8"): [1, 3],
    ("outcomes", "d10"): [10],
    ("learning", "d9"): [2, 6]
}

final_data = tokens_data.toPandas()

for key in position_checks:

    word = key[0]
    document_id = key[1]

    selected_rows = final_data[
        (final_data["term"] == word) &
        (final_data["doc_id"] == document_id)
    ]

    actual_positions = selected_rows["position"].tolist()

    if actual_positions != position_checks[key]:
        checks_passed = False
        print("Error in:", word, document_id)
        print("Expected positions:", position_checks[key])
        print("Actual positions:", actual_positions)


# 8. Print Validation Result
if checks_passed:
    print("ALL CHECKS PASSED")
else:
    print("CHECKS FAILED")


# 9. Save Output
if not os.path.exists("output"):
    os.makedirs("output")

final_data.to_csv("output/tokens.csv", index=False)

print("File saved successfully!")

# 10. Stop Spark
spark.stop()