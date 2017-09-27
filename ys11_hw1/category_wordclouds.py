import pandas as pd
import os
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

# Read data

reader = pd.read_csv("train_set.csv", delimiter="\t", quoting=3, names=["RowNum", "Id", "Title", "Content", "Category"],
                     header=0)

# Merging content per category

categories = ["Politics", "Film", "Football", "Business", "Technology"]
articles = {}
for category in categories:
    articles[category] = ""

num_of_articles = reader["Content"].size

for i in xrange(0, num_of_articles):
    articles[reader["Category"][i]] += reader["Content"][i] + " "

# Customizing stop words

stop_words = STOPWORDS.update(["said", "say", "says", "one", "now", "also", "will", "new", "year"])

# Create one workcloud per category

wordclouds = []
for category in categories:
    wordclouds.append(WordCloud("Century Gothic.ttf", stopwords=stop_words, background_color="white", width=1024,
                                height=768).generate(articles[category]))

# Export wordclouds as PNG files

if not os.path.exists("wordclouds"):
    os.makedirs("wordclouds")

for i in range(len(wordclouds)):
    plt.imshow(wordclouds[i])
    plt.axis("off")
    plt.savefig("wordclouds/" + categories[i].lower() + ".png", dpi=300)
