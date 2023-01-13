import pandas as pd

data_dict = pd.read_csv("DataDictionary.csv")
text = data_dict.iloc[:, 0:6].to_markdown()

with open("data_dictionary.md") as f:
    print(text, file=f)
