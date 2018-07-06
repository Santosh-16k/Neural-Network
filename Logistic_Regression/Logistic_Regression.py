import tensorflow as tf
import pandas as pd


#### Reading Data Set using pandas
df = pd.read_csv("Dataset/train.csv")
df = df.drop(['Ticket','Cabin'], axis=1)
df = df.dropna()

print(df)

#### Classify Data Set
