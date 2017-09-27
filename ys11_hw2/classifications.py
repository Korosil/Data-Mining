import pandas as pd
import os

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

# Read data

train = pd.read_csv('datasets/train.tsv', sep='\t', header=0)
train = train.drop('Id', 1)

# Prepare data

X = pd.get_dummies(train.drop('Label', 1))
X = X.values
y = train.Label

# 10-fold cross validation

myClassifiers = ["SVM", "Random Forest", "Naive Bayes"]
myAccuracy = {
    "SVM": {
        "clf": LinearSVC(),
        "folds": [],
        "average": None
    },
    "Random Forest": {
        "clf": RandomForestClassifier(),
        "folds": [],
        "average": None
    },
    "Naive Bayes": {
        "clf": MultinomialNB(),
        "folds": [],
        "average": None
    }
}

for classifier in myClassifiers:
    myAccuracy[classifier]["folds"] = cross_val_score(myAccuracy[classifier]["clf"], X, y, cv=10)
    myAccuracy[classifier]["average"] = myAccuracy[classifier]["folds"].mean()

# Write to CSV

results = ["Accuracy"]

for classifier in myClassifiers:
    results.append(myAccuracy[classifier]["average"])

headers = ["Statistic Measure", "SVM", "Random Forest", "Naive Bayes"]

if not os.path.exists("output"):
    os.makedirs("output")

results = pd.DataFrame([results], columns=headers)
results.to_csv("output/EvaluationMetric_10fold.csv", index=False)
