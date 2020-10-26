"""Script to analyze the Dataframes
"""
import pandas as pd
import matplotlib.pyplot as plt

german_df = pd.read_parquet('/home/felix/Desktop/Document_Scanner/text_classifier/data/german.parquet')
english_df = pd.read_parquet('/home/felix/Desktop/Document_Scanner/text_classifier/data/english.parquet')
german_df.to_csv('/home/felix/Desktop/Document_Scanner/text_classifier/data/german.csv')
english_df.to_csv('/home/felix/Desktop/Document_Scanner/text_classifier/data/english.csv')
german_df = pd.read_csv('/home/felix/Desktop/Document_Scanner/text_classifier/data/german.csv')
english_df = pd.read_csv('/home/felix/Desktop/Document_Scanner/text_classifier/data/english.csv')

print("german data")
print(german_df.info)

print("english data")
print(english_df.info)

fig = german_df[["label", "text"]].groupby("label").count().plot(kind="bar", title="German Data").get_figure()
plt.xlabel("label")
plt.ylabel("text")
plt.tight_layout()
fig.savefig('/home/felix/Desktop/Document_Scanner/text_classifier/data/de_test.pdf')

fig = english_df[["label", "text"]].groupby("label").count().plot(kind="bar", title="English Data").get_figure()
plt.xlabel("label")
plt.ylabel("text")
plt.tight_layout()
fig.savefig('/home/felix/Desktop/Document_Scanner/text_classifier/data/en_test.pdf')
