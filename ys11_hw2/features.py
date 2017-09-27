import pandas as pd
import seaborn as sns
import math
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# Calculate entropy of given data set for the target attribute

def entropy(data, target_attr):
    val_freq = {}
    data_entropy = 0.0

    for record in data:
        if record[target_attr] in val_freq:
            if record[20] in val_freq[record[target_attr]]:
                val_freq[record[target_attr]][record[20]] += 1.0
            else:
                val_freq[record[target_attr]][record[20]] = 1.0
        else:
            val_freq[record[target_attr]] = {1: 0, 2: 0}
            val_freq[record[target_attr]][record[20]] = 1.0

    for key, value in val_freq.iteritems():

        p1 = value[1] / (value[1] + value[2])
        if p1 != 0:
            h1 = -p1 * math.log(p1, 2)
        else:
            h1 = 0
        p2 = value[2] / (value[1] + value[2])
        if p2 != 0:
            h2 = -p2 * math.log(p2, 2)
        else:
            h2 = 0

        if h1 == 0:
            data_entropy += h1 * (value[1] / len(data))
        elif h2 == 0:
            data_entropy += h2 * (value[2] / len(data))
        else:
            data_entropy += (h1 + h2) * ((value[1] + value[2]) / len(data))

    return data_entropy

# Read data

train = pd.read_csv('datasets/train.tsv', sep='\t', header=0)
train = train.drop('Id', 1)

# Prepare data

X = train.drop('Label', 1)
y = train.Label
features = list(X.columns.values)

# Calculate overall entropy

val_freq = {}
overall_entropy = 0.0

for record in train.values:
    if record[list(train).index('Label')] in val_freq:
        val_freq[record[list(train).index('Label')]] += 1.0
    else:
        val_freq[record[list(train).index('Label')]] = 1.0

for freq in val_freq.values():
    overall_entropy += (-freq / len(train.values)) * math.log(freq / len(train.values), 2)

# Calculate information gain for each feature

infogain = []
for feature in features:
    infogain.append(overall_entropy - entropy(train.values, features.index(feature)))

# Calculate accuracy removing one feature in each loop

results = []
for feature, inga in zip(features, infogain):
    features.remove(feature)
    if features:
        print 'Removing feature \'' + feature + '\' with information gain: ' + str(inga)
        X = X.drop(feature, 1)
        X_train = pd.get_dummies(X)
        X_train = X_train.values
        scores = cross_val_score(RandomForestClassifier(), X_train, y, cv=10)
        results.append(scores.mean())


# Create plot

if not os.path.exists("output"):
    os.makedirs("output")

accuracy = sns.tsplot(data=results, value='Accuracy')
sns.plt.savefig('output/accuracy.svg')
