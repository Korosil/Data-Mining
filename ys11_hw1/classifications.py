import pandas as pd

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Read data

train = pd.read_csv("train_set.csv", delimiter="\t", quoting=3, names=["RowNum", "Id", "Title", "Content", "Category"],
                     header=0)

# Customize stop words

stop_words = ENGLISH_STOP_WORDS.union("said", "say", "says", "one", "now", "also", "will", "new", "year")

# Emphasize the titles of the articles
X = 10*train.Title + train.Content
y = train.Category

# 10-fold cross validation

kf = KFold(n_splits=10)

myClassifiers = ["Naive Bayes", "Random Forest", "SVM", "KNN"]
myMetrics = ["Accuracy", "Precision", "Recall", "F-Measure"]

data = {
    "Naive Bayes": {
        "folds": {
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "F-Measure": []
        },
        "average": {
            "Accuracy": 0,
            "Precision": 0,
            "Recall": 0,
            "F-Measure": 0
        }
    },
    "Random Forest": {
        "folds": {
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "F-Measure": []
        },
        "average": {
            "Accuracy": 0,
            "Precision": 0,
            "Recall": 0,
            "F-Measure": 0
        }
    },
    "SVM": {
        "folds": {
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "F-Measure": []
        },
        "average": {
            "Accuracy": 0,
            "Precision": 0,
            "Recall": 0,
            "F-Measure": 0
        }
    },
    "KNN": {
        "folds": {
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "F-Measure": []
        },
        "average": {
            "Accuracy": 0,
            "Precision": 0,
            "Recall": 0,
            "F-Measure": 0
        }
    }
}

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Convert string data to a list of vectors

    count_vect = CountVectorizer(stop_words=stop_words)
    X_train_counts = count_vect.fit_transform(X_train)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Train the classifiers

    clf_NB = MultinomialNB().fit(X_train_tfidf, y_train)
    clf_RF = RandomForestClassifier().fit(X_train_tfidf, y_train)
    clf_SVM = LinearSVC().fit(X_train_tfidf, y_train)
    clf_KNN = KNeighborsClassifier().fit(X_train_tfidf, y_train)

    # Predict the test set

    X_new_counts = count_vect.transform(X_test)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = {
        "Naive Bayes": clf_NB.predict(X_new_tfidf),
        "Random Forest": clf_RF.predict(X_new_tfidf),
        "SVM": clf_SVM.predict(X_new_tfidf),
        "KNN": clf_KNN.predict(X_new_tfidf)
    }

    # Calculate the metrics

    for classifier in myClassifiers:
        accuracy = metrics.accuracy_score(y_test, predicted[classifier])
        data[classifier]["folds"]["Accuracy"].append(accuracy)
        precision = metrics.precision_score(y_test, predicted[classifier], average="macro")
        data[classifier]["folds"]["Precision"].append(precision)
        recall = metrics.recall_score(y_test, predicted[classifier], average="macro")
        data[classifier]["folds"]["Recall"].append(recall)
        fscore = metrics.f1_score(y_test, predicted[classifier], average="macro")
        data[classifier]["folds"]["F-Measure"].append(fscore)

# Calculate the average

for classifier in myClassifiers:
    for metric in myMetrics:
        sum_average = 0
        for score in data[classifier]["folds"][metric]:
            sum_average += score

        data[classifier]["average"][metric] = sum_average / len(data[classifier]["folds"][metric])

# Write to CSV

results = []

for metric in myMetrics:
    result = []
    result.append(metric)
    for classifier in myClassifiers:
        result.append(data[classifier]["average"][metric])
    results.append(result)

headers = ["Statistic Measure", "Naive Bayes", "Random Forest", "SVM", "KNN"]

results = pd.DataFrame(results, columns=headers)
results.to_csv("EvaluationMetric_10fold.csv", index=False)

# Predict test set using optimal classifier (SVM)

test = pd.read_csv("test_set.csv", sep="\t")

data = 10*test.Title + test.Content

count_vect = CountVectorizer(stop_words=stop_words)
X_test_counts = count_vect.fit_transform(data)

tfidf_transformer = TfidfTransformer()
X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
