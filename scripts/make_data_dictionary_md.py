import pandas as pd

data_dict = pd.read_csv("../data/DataDictionary.csv")
text = data_dict.iloc[:, 0:7].to_markdown()

with open("../data_dictionary.md", 'w') as f:
    print(text, file=f)
