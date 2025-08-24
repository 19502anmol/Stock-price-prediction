import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. Data Collect करना (Apple कंपनी का पिछले 1 साल का stock data)
data = yf.download("AAPL", start="2024-01-01", end="2025-01-01")

# 2. Data देखना
print(data.head())

# 3. केवल 'Close Price' का use करेंगे
data = data[['Close']]

# 4. दिन (x-axis) बनाना
data['Day'] = np.arange(len(data))

# 5. Training Data तैयार करना
X = data[['Day']]   # Feature (Day number)
y = data['Close']   # Target (Price)

# 6. Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# 7. Prediction (अगले 30 दिन)
future_days = np.array(range(len(data), len(data)+30)).reshape(-1, 1)
future_prices = model.predict(future_days)

# 8. Visualization
plt.figure(figsize=(10,5))
plt.plot(data['Day'], data['Close'], label="Actual Prices")
plt.plot(future_days, future_prices, color='red', linestyle="dashed", label="Predicted Prices")
plt.xlabel("Days")
plt.ylabel("Stock Price ($)")
plt.legend()
plt.title("Stock Price Prediction (Apple Inc.)")
plt.show()