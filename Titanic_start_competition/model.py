import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pdb


train_data = pd.read_csv("./titanic/train.csv")

test_data = pd.read_csv("./titanic/test.csv")
test_data.head()

from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# df = pd.read_csv('train.csv')
# survived = df['Survived']


# Y_test = pd.get_dummies(test_data[id])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X)


# Calclate percentage of correct predictions if test survived==prediction survived +1 /number of ids

df = pd.read_csv('./titanic/train.csv')
survived = df['Survived']



count=0
for i in range(1,len(survived)):
  if survived[i]==predictions[i]:
    count+=1

print(f"Accuracy of the model is {count/len(survived) * 100:.3f} %")
