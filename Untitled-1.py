import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read the data from Excel file
data = pd.read_excel(r'C:\Users\BreckheiHA05\Documents\180\Restaurant Revenue.xlsx')

# Separate features (X) and target variable (y)
X = data[['Number_of_Customers', 'Menu_Price', 'Marketing_Spend', 'Average_Customer_Spending', 'Promotions', 'Reviews']]
y = data['Monthly_Revenue']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = model.score(X_test, y_test)

# Get coefficients
coefficients = model.coef_

# Print coefficients
print("Coefficients:")
for feature, coef in zip(X.columns, coefficients):
    print(f"{feature}: {coef}")

# Print accuracy
print(f"Accuracy: {accuracy}")

# Visualize the multiple regression
plt.figure(figsize=(10, 6))

# Plot features vs. actual monthly revenue
for feature in X.columns:
    plt.scatter(y_test, X_test[feature], label=feature)

# Plot the regression line
plt.plot(y_test, predictions, color='red', linewidth=2, label='Regression Line')

plt.title('Multiple Regression Analysis')
plt.xlabel('Monthly Revenue')
plt.ylabel('Features')
plt.legend()
plt.grid(True)

plt.show()


