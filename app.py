# ==============================
# 🚀 Movie Recommender Preprocessing (Docker Version)
# ==============================

from pyspark.sql import SparkSession

# ==============================
# 1. Initialize Spark
# ==============================
spark = (
    SparkSession.builder
    .appName("MovieRecommenderPreprocessing")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

print("✅ Spark Started")

# ==============================
# 2. Load Movies Dataset
# ==============================
movies = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .option("multiLine", True)
    .option("escape", '"')
    .option("quote", '"')
    .option("mode", "PERMISSIVE")
    .csv("movies_metadata.csv")
)

print("✅ Movies Loaded")

# ==============================
# 3. Clean Movies
# ==============================
from pyspark.sql.functions import col

movies = movies.filter(col("id").rlike("^[0-9]+$"))
movies = movies.withColumn("id", col("id").cast("int"))
movies = movies.dropna(subset=["title", "overview"])

print("✅ Movies Cleaned")

# ==============================
# 4. Load Other Datasets
# ==============================
credits = spark.read.csv("credits.csv", header=True, inferSchema=True)
keywords = spark.read.csv("keywords.csv", header=True, inferSchema=True)

credits = credits.withColumn("id", col("id").cast("int"))
keywords = keywords.withColumn("id", col("id").cast("int"))

print("✅ Credits & Keywords Loaded")

# ==============================
# 5. Define Schemas
# ==============================
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType

genre_schema = ArrayType(
    StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True)
    ])
)

keyword_schema = ArrayType(
    StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True)
    ])
)

cast_schema = ArrayType(
    StructType([
        StructField("cast_id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("character", StringType(), True)
    ])
)

crew_schema = ArrayType(
    StructType([
        StructField("credit_id", StringType(), True),
        StructField("job", StringType(), True),
        StructField("name", StringType(), True)
    ])
)

print("✅ Schemas Defined")

# ==============================
# 6. Parse JSON Columns
# ==============================
from pyspark.sql.functions import from_json

movies = movies.withColumn("genres_parsed", from_json(col("genres"), genre_schema))
keywords = keywords.withColumn("keywords_parsed", from_json(col("keywords"), keyword_schema))
credits = credits.withColumn("cast_parsed", from_json(col("cast"), cast_schema))
credits = credits.withColumn("crew_parsed", from_json(col("crew"), crew_schema))

print("✅ JSON Parsed")

# ==============================
# 7. UDFs
# ==============================
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

def extract_names(obj):
    if obj:
        return [i['name'].replace(" ", "") for i in obj]
    return []

def extract_top3_cast(obj):
    if obj:
        return [i['name'].replace(" ", "") for i in obj[:3]]
    return []

def extract_director(obj):
    if obj:
        for i in obj:
            if i['job'] == 'Director':
                return [i['name'].replace(" ", "")]
    return []

extract_names_udf = udf(extract_names, ArrayType(StringType()))
top3_cast_udf = udf(extract_top3_cast, ArrayType(StringType()))
director_udf = udf(extract_director, ArrayType(StringType()))

print("✅ UDFs Ready")

# ==============================
# 8. Apply UDFs
# ==============================
movies = movies.withColumn("genres_list", extract_names_udf(col("genres_parsed")))
keywords = keywords.withColumn("keywords_list", extract_names_udf(col("keywords_parsed")))
credits = credits.withColumn("cast_list", top3_cast_udf(col("cast_parsed")))
credits = credits.withColumn("director", director_udf(col("crew_parsed")))

print("✅ Features Extracted")

# ==============================
# 9. Merge DataFrames
# ==============================
df = movies.join(credits, on="id").join(keywords, on="id")

print("✅ Data Merged")

# ==============================
# 10. Select Columns
# ==============================
df = df.select(
    "id",
    "title",
    "overview",
    "genres_list",
    "cast_list",
    "director",
    "keywords_list"
)

print("✅ Columns Selected")

# ==============================
# 11. Create Tags
# ==============================
from pyspark.sql.functions import concat_ws

df = df.withColumn(
    "tags",
    concat_ws(
        " ",
        col("overview"),
        concat_ws(" ", col("genres_list")),
        concat_ws(" ", col("cast_list")),
        concat_ws(" ", col("director")),
        concat_ws(" ", col("keywords_list"))
    )
)

print("✅ Tags Created")

# ==============================
# 12. Final Dataset
# ==============================
df_final = df.select("id", "title", "tags")
df_final = df_final.dropna(subset=["tags"])

print("✅ Final Dataset Ready")

# ==============================
# 13. Show Output
# ==============================
df_final.show(5, truncate=False)

# ==============================
# 14. Save Output
# ==============================
df_final.write.mode("overwrite").parquet("processed_movies")

print("🎉 Preprocessing Completed Successfully!")