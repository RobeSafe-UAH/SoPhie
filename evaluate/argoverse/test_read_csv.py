import pandas as pd
import numpy as np
import pdb

filename = "/home/robesafe/shared_home/exp9_metrics_val_25_percent.csv"
df = pd.read_csv(filename,sep=" ")

n = 2
df.drop(df.tail(n).index, inplace=True) # Remove last two rows

pdb.set_trace()