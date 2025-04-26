import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"
df = pd.read_csv(url)
print(df.head())

X = df[['TV', 'radio', 'newspaper']]
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

plt.scatter(X_test['TV'], y_test, color='blue', label='Actual Sales')
plt.scatter(X_test['TV'], y_pred, color='red', label='Predicted Sales')

real_time_data = pd.DataFrame([{
    'TV':230.1,
    'radio':37.8,
    'newspaper':69.2
}])
real_time_prediction = model.predict(real_time_data)
print("predicted sales for realtime data :",round(real_time_prediction[0],2))
