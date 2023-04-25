import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the training and test data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Initial data exploration
train_df.head()
train_df.info()
train_df.describe()
sns.countplot(x='Survived', data=train_df)

# Data preprocessing
train_df.drop(columns=['Cabin'], inplace=True)
test_df.drop(columns=['Cabin'], inplace=True)
train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
test_df['Age'].fillna(test_df['Age'].mean(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)

# Feature engineering
train_df['Title'] = train_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
test_df['Title'] = test_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
train_df['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Jonkheer', 'Capt'], 'Rare', inplace=True)
train_df['Title'].replace(['Ms', 'Mlle'], 'Miss', inplace=True)
test_df['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Jonkheer', 'Capt'], 'Rare', inplace=True)
test_df['Title'].replace(['Ms', 'Mlle'], 'Miss', inplace=True)

# Select the relevant features and target variable
X_train = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']]
y_train = train_df['Survived']
X_test = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']]

# One-hot encode the categorical features
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Create a random forest classifier and fit it on the training data
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Use the trained model to make predictions on the test data
y_pred = rf.predict(X_test)

# Save the predictions in a CSV file
submission_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_pred})
submission_df.to_csv('submission.csv', index=False)
