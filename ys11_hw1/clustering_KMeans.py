import pandas as pd
import nltk

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from nltk.cluster.kmeans import KMeansClusterer

NUM_CLUSTERS = 5

# Read data

reader = pd.read_csv("train_set.csv", delimiter="\t", quoting=3, names=["RowNum", "Id", "Title", "Content", "Category"],
                     header=0)

# Customize stop words

stop_words = ENGLISH_STOP_WORDS.union("said", "say", "says", "one", "now", "also", "will", "new", "year")

# Convert string data to a list of vectors

data = []
for i in range(reader["Content"].size):
    data.append(reader["Title"][i] + " " + reader["Content"][i])

vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(data)
transformer = TfidfTransformer()
X = transformer.fit_transform(X)
svd = TruncatedSVD(n_components=40)
X = svd.fit_transform(X)

# Create clusters

kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
clusters = kclusterer.cluster(X, assign_clusters=True)

# Calculate percentage of each category in clusters

categories = ["Politics", "Film", "Football", "Business", "Technology"]
num_of_articles_per_cluster = {}

for category in categories:
    num_of_articles_per_cluster[category] = [0, 0, 0, 0, 0]

row = 0
for i in clusters:
    num_of_articles_per_cluster[reader["Category"][row]][i] += 1
    row += 1

frequencies = []
for i in range(5):
    frequency_of_category_per_cluster = []

    sum_of_documents = 0
    for category in categories:
        sum_of_documents += num_of_articles_per_cluster[category][i]

    for category in categories:
        frequency_of_category_per_cluster.append(
            round(float(num_of_articles_per_cluster[category][i]) / float(sum_of_documents), 4))

    frequencies.append(frequency_of_category_per_cluster)

# Write to CSV

results = []
for i in range(NUM_CLUSTERS):
    frequencies[i].insert(0, "Cluster " + str(i))
    results.append(frequencies[i])

categories.insert(0, "Clusters")

results = pd.DataFrame(results, columns=categories)
results.to_csv("clustering_KMeans.csv", index=False)
