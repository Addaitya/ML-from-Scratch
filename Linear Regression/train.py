from Linear_regression import Linear_regression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = fetch_california_housing(return_X_y=True)
# Note normalising your data is important because the value of variables may overflow(i.e become infinity).
X_norm = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, random_state=42)
reg = Linear_regression()
w_final, b_final = reg.fit(X_train, y_train)

print(f"Squared error for training data: {reg.compute_squared_error(X_train, y_train)}")
print(f"Squared error for test data: {reg.compute_squared_error(X_test, y_test)}")
print(f"test score: {reg.score(X_train, y_train)}")