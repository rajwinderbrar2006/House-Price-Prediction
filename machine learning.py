import pandas as pd
data=pd.read_csv("HOUSE PRICE PREDICATION")
print(data)

print(data.head(33))
print(data.tail())
print(data.info())
print(data.describe())
#check missing values
print(data.isnull().sum())
#check duplicates values
print(data.isnull().sum())
#handle text
data=data.drop("distance_km",axis=1)
print(data)
data=data.dropna()
print(data)
print(data.isnull().sum())

#fillna()with value
data=data.fillna(0)
print(data)
print(data.isnull().sum())

#fillna with mean,mode,median
# mode method
data["bedrooms"]=data["bedrooms"].fillna(data["bedrooms"].mode())
print(data.isnull().sum())
# mean methond
data["age_years"]=data["age_years"].fillna(data["age_years"].mean())
print(data.isnull().sum())
#split data into features and target
# Split data into features and target
X = data.drop("price_lakh", axis=1)
Y = data["price_lakh"]

# Split data into training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=True, random_state=42
)
print(X_train)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#step 7: model training 
#regression-linear regression
#classification logistic regression

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)

# Model testing
y_predict = model.predict(X_test)
print(y_predict)

# MSE
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_predict)
print(mse)

from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_test,y_predict)
print(mae)

#classification
#accuracy 
from sklearn.metrics import r2_score

accuracy = r2_score(y_test, y_predict)
print("R2 Score:", accuracy)