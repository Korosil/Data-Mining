import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier

# Read train set

train = pd.read_csv('datasets/train.tsv', sep='\t', header=0)
train = train.drop('Id', 1)

# Prepare train set

X_train = train.drop('Label', 1)
X_train = X_train.drop(['Attribute18', 'Attribute11', 'Attribute19', 'Attribute16', 'Attribute17'], 1)
X_train = pd.get_dummies(X_train)
X_train = X_train.values
y_train = train.Label

# Train classifier

clf = RandomForestClassifier().fit(X_train, y_train)

# Read test set

test = pd.read_csv('datasets/test.tsv', sep='\t', header=0)

# Prepare test set

X_test = test.drop('Id', 1)
X_test = X_test.drop(['Attribute18', 'Attribute11', 'Attribute19', 'Attribute16', 'Attribute17'], 1)
X_test = pd.get_dummies(X_test)
X_test = X_test.values

# Predict

predicted = clf.predict(X_test)

# Write to CSV

results = []
for client, label in zip(test.Id, predicted):
    row = []
    if label == 1:
        label = 'Good'
    else:
        label = 'Bad'
    row.append(client)
    row.append(label)
    results.append(row)

headers = ["Client_ID", "Predicted_Label"]

if not os.path.exists("output"):
    os.makedirs("output")

results = pd.DataFrame(results, columns=headers)
results.to_csv("output/testSet_Predictions.csv", index=False, sep="\t")
