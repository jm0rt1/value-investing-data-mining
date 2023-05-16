import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set the stock symbol and date range
symbol = 'AAPL'
start_date = '2010-01-01'
end_date = '2021-12-31'

# Fetch historical stock price data
stock_data = yf.download(symbol, start=start_date, end=end_date)

# Calculate technical indicators
stock_data['SMA10'] = stock_data['Adj Close'].rolling(window=10).mean()
stock_data['SMA50'] = stock_data['Adj Close'].rolling(window=50).mean()
stock_data['EMA10'] = stock_data['Adj Close'].ewm(span=10, adjust=False).mean()
stock_data['EMA50'] = stock_data['Adj Close'].ewm(span=50, adjust=False).mean()
stock_data['RSI'] = 100 - \
    (100 /
     (1 + (stock_data['Adj Close'].diff(1).fillna(0) > 0).rolling(14).mean()))

# Define the target variable
stock_data['Target'] = (stock_data['Adj Close'].shift(-1)
                        > stock_data['Adj Close'])

# Drop rows with missing data
stock_data.dropna(inplace=True)

# Split the data into training and testing sets
X = stock_data[['SMA10', 'SMA50', 'EMA10', 'EMA50', 'RSI']]
y = stock_data['Target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Build the machine learning model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2%}')
