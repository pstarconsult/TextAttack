import datasets
import pandas as pd
from sklearn.model_selection import train_test_split

dataset_df = pd.read_csv("zh_sentiment.tsv", sep="\t", index_col=False)
del dataset_df["Unnamed: 0"]
dataset_df.columns = ["text", "label"]

train, test = train_test_split(dataset_df, test_size=0.2, shuffle=True, stratify=dataset_df["label"])

train.to_csv("zh_sentiment_train.tsv", sep="\t", index=False)
test.to_csv("zh_sentiment_test.tsv", sep="\t", index=False)

