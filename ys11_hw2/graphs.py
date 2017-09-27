import pandas as pd
import seaborn as sns
import os

# Read data

reader = pd.read_csv('datasets/train.tsv', sep='\t', header=0)
reader = reader.drop('Id', 1)

# Create output directory

if not os.path.exists("output"):
    os.makedirs("output")

# Create histograms and box plots for every feature

for i, attribute in enumerate(reader.drop('Label', 1)):
    if reader[attribute].dtype == "object":
        sns.plt.figure(i)
        g = sns.countplot(y=attribute, hue="Label", data=reader, palette="Reds_d")
        g.axes.set_title(attribute + " histogram", fontsize=18)
        sns.plt.savefig('output/' + attribute + '.svg')
    else:
        sns.plt.figure(i)
        g = sns.boxplot(y=attribute, x="Label", data=reader)
        g.axes.set_title(attribute + " box plot", fontsize=18)
        sns.plt.savefig('output/' + attribute + '.svg')
