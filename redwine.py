import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pandas.plotting import scatter_matrix
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from joblib import dump

# Load the data from the CSV file
file_path = 'C:\\Users\\mh183\\OneDrive\\Documents\\redwine.csv'
data = pd.read_csv(file_path)

# Print the first 5 data
print(data.head())

# Print the information
print(data.info())

# Print the data description
print(data.describe())


# Split the data into train_set and test_set
train_set, test_set = train_test_split(data, test_size=0.2, random_state=80)

# Print the number of rows in train_set and test_set
print(f"Rows in train_set: {len(train_set)}\nRows in test_set: {len(test_set)}")

# Compute the correlation matrix
corr_matrix = data.corr()
print(corr_matrix['quality'].sort_values(ascending=False))

# Plot the scatter matrix
columns_to_access = ['fixed acidity', 'volatile acidity', 'citric acid', 'density', 'pH', 'alcohol', 'quality']
scatter_matrix(data[columns_to_access], figsize=(20, 9))
# plt.show()

# Separate features and labels in the train_set
x_train = train_set.drop("quality", axis=1)
y_train = train_set["quality"].copy()

# Create a data preprocessing pipeline
my_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

# Fit the pipeline and transform the training data
X_train_preprocessed = my_pipeline.fit_transform(x_train)

# Create and train the Linear Regression model
model = RandomForestRegressor()
model.fit(X_train_preprocessed, y_train)

# Some example data to predict
some_data = x_train.iloc[:5]
some_labels = y_train.iloc[:5]

# Preprocess the example data and make predictions
predict_data = my_pipeline.transform(some_data)
predictions = model.predict(predict_data)
print("Predictions:", predictions)

print(some_labels)

model_prediction = model.predict(X_train_preprocessed)
mse = mean_squared_error(y_train, model_prediction)
qrt = np.sqrt(mse)
print(qrt)

score = cross_val_score(model, X_train_preprocessed, y_train, scoring="neg_mean_squared_error", cv=10)
qrt_score = np.sqrt(-score)
print(qrt_score)

dump(model, 'redwine.joblib')