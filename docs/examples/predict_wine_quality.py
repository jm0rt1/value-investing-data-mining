"""
    In this example, we will use the Wine Quality dataset to
    predict the quality of wine based on several
    chemical attributes. The dataset contains 1599 
    instances and 12 attributes. We will preprocess
    the data, split it into training and test sets, train a random forest model,
    and evaluate its performance using
    mean squared error (MSE) and R-squared score.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('winequality.csv', delimiter=';')

# Split the dataset into features and target variable
X = df.drop(columns=['quality'])
y = df['quality']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Perform data preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a random forest model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean squared error: {mse}')
print(f'R-squared score: {r2}')
