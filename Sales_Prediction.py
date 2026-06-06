import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

#Load Data
df = pd.read_csv(r"C:\Users\Lakshmi Anusha\Downloads\archive.zip")

#Display first five rows
print("\nFirst 5 rows:")
print(df.head())

#Features 
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

#split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Create model
lin_reg = LinearRegression()

#Train model
lin_reg.fit(X_train, y_train)

#predict sales
y_pred_lr = lin_reg.predict(X_test)

#Print predictions
print("\nPredictions sales:")
print(y_pred_lr)

#model evaluation
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print("Mean Absolute Error:",mae_lr)
print("R2 score:",r2_lr)

#Plotting
best_name="Linear Regression"
plt.scatter(df['TV'],df['Sales'])
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title(f"Actual vs Predicted Sales ({best_name})")
plt.show()

