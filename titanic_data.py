import matplotlib.pyplot as plt
import pandas as pd

train_df = pd.read_csv('data/train.csv')
print(train_df.value_counts())
test_df = pd.read_csv('data/test.csv')

train_df['Age'].fillna(value=train_df['Age'].median(), inplace=True)
train_df['Survived'].value_counts().plot(kind='bar')
plt.show()
