#Cement – quantitative – kg in a m3 mixture
#Blast Furnace Slag – quantitative – kg in a m3 mixture
#Fly Ash – quantitative – kg in a m3 mixture
#Water – quantitative – kg in a m3 mixture
#Superplasticizer – quantitative – kg in a m3 mixture
#Coarse Aggregate – quantitative – kg in a m3 mixture
#Fine Aggregate – quantitative – kg in a m3 mixture
#Age – quantitative – Day (1~365)
#Concrete compressive strength – quantitative – MPa

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

pd.set_option('display.max_columns', None)

data = pd.read_csv("concrete.csv")

total_weight = []

weight = 0

for i in range(1030):
    for item in data:
        weight += data[item][i]
    total_weight.append(weight)
    weight = 0
    
cement_percentage = []
slag_percentage = []
ash_percentage = []
water_percentage = []
superplastic_percentage = []
coarseagg_percentage = []
fineagg_percentage = []

percentage = 0

for i in range(1030):
    percentage = round(data['cement'][i]/total_weight[i],4)*100
    cement_percentage.append(percentage)
    percentage = round(data['slag'][i]/total_weight[i],4)*100
    slag_percentage.append(percentage)
    percentage = round(data['ash'][i]/total_weight[i],4)*100
    ash_percentage.append(percentage)
    percentage = round(data['water'][i]/total_weight[i],4)*100
    water_percentage.append(percentage)
    percentage = round(data['superplastic'][i]/total_weight[i],4)*100
    superplastic_percentage.append(percentage)
    percentage = round(data['coarseagg'][i]/total_weight[i],4)*100
    coarseagg_percentage.append(percentage)
    percentage = round(data['fineagg'][i]/total_weight[i],4)*100
    fineagg_percentage.append(percentage)
    
data_percentage = data.copy()
    
data_percentage["cement"] = cement_percentage
data_percentage["slag"] = slag_percentage
data_percentage["ash"] = ash_percentage
data_percentage["water"] = water_percentage
data_percentage["superplastic"] = superplastic_percentage
data_percentage["coarseagg"] = coarseagg_percentage
data_percentage["fineagg"] = fineagg_percentage

pd.set_option('display.width', 1000)

print(data_percentage.head())
print(data_percentage.tail())


sns.pairplot(data_percentage,size = 2.5)

plt.tight_layout()

plt.show()

corMat = pd.DataFrame(data_percentage.corr())
fig, ax = plt.subplots(figsize=(10,10))   

sns.heatmap(corMat,annot = True)

X = []

y = []

for i in range(1030):
    temp = []
    for item in data_percentage:
        if item != "strength":
            temp.append(data_percentage[item][i])
        else:
            y.append(data_percentage[item][i])
    X.append(temp)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
all_reg = LinearRegression()

# Fit the regressor to the training data
all_reg.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred = all_reg.predict(X_test)

print("Here is the linear regression stat data:")

print("coef: ",all_reg.coef_)
print("intercept: ",all_reg.intercept_)

# Compute and print R^2 and RMSE
print("R^2: {}".format(round(all_reg.score(X_test, y_test),4)))
rmse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: {}".format(round(rmse,4))+"\n")

fig, ax = plt.subplots(figsize=(10,10))   

plt.scatter(y_pred, y_pred - y_test,  color='red', s = 2)
plt.hlines(y = 0, xmin = 0, xmax = 80, color = 'black')

plt.xlabel("Test sample for strength in MPa")
plt.ylabel("Sesidual errors")
        
plt.show()

test_array = [0.00001,0.001,1,10,100]

for i in test_array:
    # Create the regressor: reg_all
    ridge = Lasso(alpha = i)
    
    # Fit the regressor to the training data
    ridge.fit(X_train,y_train)
    
    # Predict on the test data: y_pred
    y_pred = ridge.predict(X_test)
    
    print("Here is the Lasso regression with alpha = ",i," stat data:")
    
    print("coef: ",ridge.coef_)
    print("intercept: ",ridge.intercept_)
    
    # Compute and print R^2 and RMSE
    print("R^2: {}".format(round(ridge.score(X_test, y_test),4)))
    rmse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error: {}".format(round(rmse,4))+"\n")
    
    fig, ax = plt.subplots(figsize=(10,10))   
    
    plt.scatter(y_pred, y_pred - y_test,  color='red', s = 2)
    plt.hlines(y = 0, xmin = 0, xmax = 80, color = 'black')
    
    plt.xlabel("Test sample for strength in MPa")
    plt.ylabel("Residual errors")
            
    plt.show()
    
pic = []
    
test = [0.1,0.3,0.5,0.7,0.9]
    
fig, axarr = plt.subplots(5,5,figsize=(15,15),sharex=True, sharey=True )


a = int(0)

b = int(0)

for i in test_array:
    for j in test:
        ridge = ElasticNet(alpha = i, l1_ratio = j)
        
        # Fit the regressor to the training data
        ridge.fit(X_train,y_train)
        
        # Predict on the test data: y_pred
        y_pred = ridge.predict(X_test)
        
        print("Here is the ElasticNet regression with alpha = ",i," and L1 ratio = ",j," stat data:")
        
        print("coef: ",ridge.coef_)
        print("intercept: ",ridge.intercept_)
        
        # Compute and print R^2 and RMSE
        print("R^2: {}".format(round(ridge.score(X_test, y_test),4)))
        rmse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error: {}".format(round(rmse,4))+"\n")
        
        axarr[a, b].set_title("alpha:"+str(i)+" L1 ratio:"+str(j))
        
        axarr[a, b].scatter(y_pred, y_pred - y_test,  color='red', s = 2)
        axarr[a, b].hlines(y = 0, xmin = 0, xmax = 80, color = 'black')
        
        b+=1
    a+=1
    b=0
    
print("My name is {Xiaokang Feng}")

print("My NetID is: {668129807}")

print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
    
