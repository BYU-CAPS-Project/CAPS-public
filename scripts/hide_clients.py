import pandas as pd
import numpy as np
from random import sample, seed
import sys
import os.path


data_path = ["..", "data"]

apts_df = pd.read_csv(os.path.join(*data_path, "Appointments.csv"), low_memory=False)
apts_df = apts_df.replace('#NULL!', np.nan)
apts_df.dropna(subset=["ClientID"], inplace=True)
apts_df["ClientID"] = apts_df["ClientID"].astype(float).astype(int).astype(str)

seed(34)
hide = sample(sorted(apts_df['ClientID'].unique()), int(len(apts_df['ClientID'].unique()) * .2))

pd.Series(hide).to_csv(os.path.join(*data_path, 'ClientsHiddenAway'), index=False)

print(f"""seed: 34
python: {sys.version}
      """,)

"""
seed: 34
python: 3.10.10 (tags/v3.10.10:aad5f6a, Feb  7 2023, 17:20:36) [MSC v.1929 64 bit (AMD64)]
"""