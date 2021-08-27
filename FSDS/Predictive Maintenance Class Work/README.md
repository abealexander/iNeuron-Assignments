# Predictive-Maintenance-Class-Work
Analysis:
UDI and Product ID are unique values and will not contribute to our model. We will drop them.

Type is a categorical column with three distinct values L, M and H.

Air temperature [K] is our target variable.

Rotational speed [rpm] has a Left Skewed Normal Distribution.

Torque [Nm] has a Normal Distribution.

Machine failure, TWF, HDF, PWF, OSF, RNF all have Boolean values which are denote 1 for failure.

There are no missing values in the Dataset.

We can see high correlation between Air temperature (target variable) and Process temperature (feature).

There seems to exist multicollinearity i.e. relationship between independent features in the case of Machine Failure and TWF, HDF, PWF, OSF, RNF. Also in the case of Rotational Speed and Torque.

R2 Score: 0.7704937194618728

Test Case Prediction: [303.27405982]

R2 Score without Normalization: 0.7717468080971295

R2 Score without fixing Multicollinearity: 0.7718438983900406
