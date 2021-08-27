# Importing Libraries
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import r2_score

# Loading Data
df = pd.read_csv('ai4i2020.csv')

# Pandas Profiling
pf = df.profile_report()
pf.to_file(output_file="Dataset_Report.html")

# Data Analysis
print("UDI and Product ID are unique values and will not contribute to our model. We will drop them.\n",
      "Type is a categorical column with three distinct values L, M and H.\n",
      "Air temperature [K] is our target variable.\n",
      "Rotational speed [rpm] has a Left Skewed Normal Distribution.\n",
      "Torque [Nm] has a Normal Distribution.\n",
      "Machine failure, TWF, HDF, PWF, OSF, RNF all have Boolean values which are denote 1 for failure.\n",
      "There are no missing values in the Dataset.\n",
      "We can see high correlation between Air temperature (target variable) and Process temperature (feature).\n",
      "There seems to exist multicollinearity i.e. relationship between independent features in the case of Machine Failure and TWF, HDF, PWF, OSF, RNF. Also in the case of Rotational Speed and Torque.")

# Dropping Columns
df.drop(['UDI', 'Product ID'], axis=1, inplace=True)

# Checking for NaN values
print(df.isnull().values.any())

# Label Encoding
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])

# Check Multicollinearity
def calc_vif(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)
print(calc_vif(df.drop('Air temperature [K]', axis=1)))

# Fixing Multicollinearity
df1 = df.drop('Rotational speed [rpm]', axis=1)
print(calc_vif(df1.drop('Air temperature [K]', axis=1)))

# Normalize the data
X = df1.drop('Air temperature [K]', axis=1)
y = df1['Air temperature [K]']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=27)
X_train_norm = MinMaxScaler().fit_transform(X_train)
X_test_norm = MinMaxScaler().fit_transform(X_test)

# Build a Model
lr = LinearRegression()
lr.fit(X_train_norm, y_train)

# Save Model
filename = 'model.sav'
pickle.dump(lr, open(filename, 'wb'))

# Model Accuracy
y_pred = lr.predict(X_test_norm)
print("R2 Score:", r2_score(y_test, y_pred))

# Test Case
Test_Case = [[0.5, 0.87654321, 0.37087912, 0.04743083, 0, 0, 0, 0, 0, 0]]
print("Test Case Prediction:",lr.predict(Test_Case))

# Checking Model Accuracy without Normalization
lr1 = LinearRegression()
lr1.fit(X_train, y_train)
y_pred = lr1.predict(X_test)
print("R2 Score without Normalization:", r2_score(y_test, y_pred))

# Checking Model Accuracy without fixing multicollinearity
X = df.drop('Air temperature [K]', axis=1)
y = df['Air temperature [K]']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=27)
lr2 = LinearRegression()
lr2.fit(X_train, y_train)
y_pred = lr2.predict(X_test)
print("R2 Score without fixing Multicollinearity:", r2_score(y_test, y_pred))
