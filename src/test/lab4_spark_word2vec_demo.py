import gzip
import json
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import lower, regexp_replace, split, col
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

spark = SparkSession.builder \
    .appName("SparkWord2VecDemo") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

def read_json_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "text" in obj and obj["text"].strip():
                    yield obj["text"]
            except:
                continue

input_path = r"D:\ScalaProjects\nlp\spark_labs\data\c4-train.00000-of-01024-30K.json.gz"

rdd = spark.sparkContext.parallelize(read_json_gz(input_path))
df = rdd.map(lambda t: (t, )).toDF(["text"])

print(f"Loaded {df.count()} rows from {input_path}")

df = df.withColumn("text", lower(col("text")))
df = df.withColumn("text", regexp_replace(col("text"), "[^a-zA-Z\\s]", ""))
df = df.withColumn("words", split(col("text"), "\\s+"))
df = df.filter(col("words").getItem(0).isNotNull())

word2Vec = Word2Vec(vectorSize=100, minCount=5, inputCol="words", outputCol="result")
model = word2Vec.fit(df)
print("Word2Vec model trained successfully!")

try:
    synonyms = model.findSynonyms("computer", 5)
    print("\nMost similar words to 'computer':")
    synonyms.show(truncate=False)
except Exception as e:
    print(f"Could not find synonyms for 'computer': {e}")

model.write().overwrite().save("/content/spark_word2vec_model")
print("Model saved to /content/spark_word2vec_model")

vocab = model.getVectors().sample(False, 0.01, seed=42).toPandas()  # sample 1% vocab để tránh OOM
vectors = vocab['vector'].apply(lambda v: v.toArray()).tolist()

pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

vocab['x'] = reduced[:, 0]
vocab['y'] = reduced[:, 1]

sample = vocab.sample(50, random_state=42)

plt.figure(figsize=(10, 8))
plt.scatter(sample['x'], sample['y'], alpha=0.6)
for i, word in enumerate(sample['word']):
    plt.annotate(word, (sample['x'].iloc[i], sample['y'].iloc[i]), fontsize=9)

plt.title("PCA Visualization of Spark Word2Vec Embeddings")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.grid(True)
plt.show()

spark.stop()