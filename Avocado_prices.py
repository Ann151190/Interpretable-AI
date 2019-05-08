# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:01:44 2019

@author: aneogy
"""

# -*- coding: utf-8 -*-
"""
Created on 1st March 2019

@author: Ananya
"""
import os
import numpy as np
import pandas as pd
import sklearn as sk
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

os.chdir("C:/Users/aneogy/Desktop/ML-Practise")
data_ori = pd.read_csv('avocado.csv')

# How does the data look?

print(data_ori.shape)
data_ori.columns

#Altering the display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Understanding the description of the dataset
print(data_ori.describe())
pd.isnull(data_ori).sum()

#Dropping unusuable columns
data_ori1 = data_ori.drop(['Unnamed: 0'], axis = 1)
data_ori1.columns

# convert Date column's format;

data_ori1['Date'] =pd.to_datetime(data_ori1.Date)
data_ori1.sort_values(by=['Date'], inplace=True, ascending=True)
data_ori1.head()

##################################################################
###################### EDA #######################################
##################################################################

#Heat map
f, ax = plt.subplots(1, 1, figsize=(10,8))
corr = data_ori1.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax)
ax.set_title("Correlation Matrix", fontsize=14)
plt.show()

#Distribution of the avocado prices

plt.figure(figsize=(12,5))
plt.title("Distribution Price")
ax = sns.distplot(data_ori1["AveragePrice"], color = 'b')

#Boxplot of the avocado prices by type

sns.boxplot(y="type", x="AveragePrice", data=data_ori1)

#Boxplot of the avocado prices by year

fig, ax = plt.subplots(1, 1, figsize=(10,8))
sns.boxplot(x='year',y='AveragePrice',data=data_ori1,color='blue')
#sns.boxplot(x='region', y ='AveragePrice',data=data_ori1,color='red')

#Trend Analysis of Average price of avocados over the years and for each type
# Average price of Conventional Avocados over time

mask = data_ori1['type']== 'conventional'
plt.rc('figure', titlesize=50)
fig = plt.figure(figsize = (30, 10))
fig.suptitle('Average Price of Conventional Avocados Over Time', fontsize=25)
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.93)

dates = data_ori1[mask]['Date'].tolist()
avgPrices = data_ori1[mask]['AveragePrice'].tolist()

plt.scatter(dates, avgPrices, c=avgPrices, cmap='plasma')
ax.set_xlabel('Date',fontsize = 15)
ax.set_ylabel('Average Price (USD)', fontsize = 15)
plt.show()


# Average price of Conventional Avocados over time
mask = data_ori1['type']== 'organic'
plt.rc('figure', titlesize=50)
fig = plt.figure(figsize = (30, 10))
fig.suptitle('Average Price of Conventional Avocados Over Time', fontsize=25)
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.93)

dates = data_ori1[mask]['Date'].tolist()
avgPrices = data_ori1[mask]['AveragePrice'].tolist()

plt.scatter(dates, avgPrices, c=avgPrices, cmap='plasma')
ax.set_xlabel('Date',fontsize = 15)
ax.set_ylabel('Average Price (USD)', fontsize = 15)
plt.show()

# Time Series Analysis
# Analysing time series data 

data_ori2 = data_ori1[['Date', 'AveragePrice']]
data_ori2 = data_ori2.set_index('Date')

weekly_df = data_ori2.resample('W').mean()
w_df = weekly_df.reset_index().dropna()

w_df.sort_values(by=['Date'])
w_df.head()

#See how the average prices per week varies by month over the years 2015-2018
import matplotlib.dates as mdates
fig = plt.figure(figsize = (30, 10))
ax = plt.axes()

#set ticks every month
ax.xaxis.set_major_locator(mdates.MonthLocator())

#set major ticks format
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.plot(w_df['Date'],w_df['AveragePrice'],color='r', linewidth=5)
plt.xlabel("2015-2018", fontsize = 15)
plt.ylabel("Avocado Price USD", fontsize = 15)
plt.legend()
plt.show()

# Feature Importance



###################################################################
###################   LIME ########################################
###################################################################



#standardizing of data and splitting the datasets
X = x.drop(['subscribe_code'], axis =1)
Y = data_ori['y']
        
evens = [n for n in range(X_ori.shape[0]) if n % 2 == 0]
X_train = X.iloc[evens,:]
Y_train = Y.iloc[evens]
#print(Y_train.value_counts())


odds = [n for n in range(X_ori.shape[0]) if n % 2 != 0]
X_test = X.iloc[odds,:]
Y_test = Y.iloc[odds]
#print(Y_test.value_counts())

#create report dataframe
report = pd.DataFrame(columns=['Model','Acc.Train','Acc.Test'])


#######################
# k-nearest neighbour #
#######################

from sklearn.neighbors import KNeighborsClassifier
knnmodel = KNeighborsClassifier(n_neighbors=13)
knnmodel.fit(X_train, Y_train)

Y_train_pred = knnmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = knnmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)

#find optimal k
accuracies = []
for k in range(1, 21):
    knnmodel = KNeighborsClassifier(n_neighbors=k)
    knnmodel.fit(X_train, Y_train)
    Y_test_pred = knnmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    print(k, accte)
    accuracies.append(accte)
plt.plot(range(1, 21), accuracies)
plt.xlim(1,20)
plt.xticks(range(1, 21))
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracies')
plt.show()
opt_k = np.argmax(accuracies) + 1
print('Optimal k =', opt_k)

#set the value of value of k to it optimum and repeat the analysis
knnmodel = KNeighborsClassifier(n_neighbors=opt_k)
knnmodel.fit(X_train, Y_train)
Y_train_pred = knnmodel.predict(X_train)
acctr = accuracy_score(Y_train, Y_train_pred)
Y_test_pred = knnmodel.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['k-NN', acctr, accte]

################
# Naive Bayes #
###############

#model creation 
from sklearn.naive_bayes import GaussianNB
nbmodel = GaussianNB()
nbmodel.fit(X_train, Y_train)

#model applied to training and test data
Y_train_pred = nbmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)

Y_test_pred = nbmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Naive Bayes', acctr, accte]


#######################
# Logistic Regression #
#######################

#logistic regression is implemented
from sklearn.linear_model import LogisticRegression
lrmodel = LogisticRegression()
lrmodel.fit(X_train, Y_train)

#model applied to the training and test data
Y_train_pred = lrmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)

Y_test_pred = lrmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Logistic Regression', acctr, accte]

######################
#   Decision Trees   #
######################

#Simple decision tree model is built where it supports both Gini and Entropy#
#but takes Gini as default#
from sklearn.tree import DecisionTreeClassifier
etmodel = DecisionTreeClassifier(criterion='entropy', random_state=0)
etmodel.fit(X_train, Y_train)

#Model is applied to both traing and test data
Y_train_pred = etmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = etmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)

#find optimal max_depth which determines the max depth of a branch.
#Can be interpreted as prepruning
accuracies = np.zeros((2,20), float)
for k in range(0, 20):
    etmodel = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=k+1)
    etmodel.fit(X_train, Y_train)
    Y_train_pred = etmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[0,k] = acctr
    Y_test_pred = etmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[1,k] = accte
plt.plot(range(1, 21), accuracies[0,:])
plt.plot(range(1, 21), accuracies[1,:])
plt.xlim(1,20)
plt.xticks(range(1, 21))
plt.xlabel('Max_depth')
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracies (Entropy)')
plt.show()

#use the optimal max depth to re-train the model and perform tests
etmodel = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=4)
etmodel.fit(X_train, Y_train)
Y_train_pred = etmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = etmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Tree (Entropy)', acctr, accte]
'''
#show tree using graphviz
import graphviz 
dot_data = sk.tree.export_graphviz(etmodel, out_file=None,
                         feature_names=list(X),  
                         filled=True, rounded=True,  
                         special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.format = 'png'
graph.render("bank_entropy") 
'''

#    Criterion parameter =  Gini      #

#model built
from sklearn.tree import DecisionTreeClassifier
gtmodel = DecisionTreeClassifier(random_state=0)
gtmodel.fit(X_train, Y_train)

#model used in carrying out training and test
Y_train_pred = gtmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = gtmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)

##find optimal max_depth which determines the max depth of a branch.
#Can be interpreted as prepruning
accuracies = np.zeros((2,20), float)
for k in range(0, 20):
    gtmodel = DecisionTreeClassifier(random_state=0, max_depth=k+1)
    gtmodel.fit(X_train, Y_train)
    Y_train_pred = gtmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[0,k] = acctr
    Y_test_pred = gtmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[1,k] = accte
plt.plot(range(1, 21), accuracies[0,:])
plt.plot(range(1, 21), accuracies[1,:])
plt.xlim(1,20)
plt.xticks(range(1, 21))
plt.xlabel('Max_depth')
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracies (Gini)')
plt.show()

#use the optimal max depth to re-train the model and perform tests
gtmodel = DecisionTreeClassifier(random_state=0, max_depth=4)
gtmodel.fit(X_train, Y_train)
Y_train_pred = gtmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = gtmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Tree (Gini)', acctr, accte]

'''
import graphviz 
dot_data = sk.tree.export_graphviz(gtmodel, out_file=None,
                         feature_names=list(X),  
                         filled=True, rounded=True,  
                         special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.format = 'png'
graph.render("Churn_gini")'''

#show feature importance
list(zip(X, gtmodel.feature_importances_))
index = np.arange(len(gtmodel.feature_importances_))
bar_width = 1.0
plt.bar(index, gtmodel.feature_importances_, bar_width)
plt.xticks(index,  list(X), rotation=90) # labels get centered
plt.show()

#################
# Random Forest #
#################

#Model is built
from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier(random_state=0)
rfmodel.fit(X_train, Y_train)

#Model used on training and test data
Y_train_pred = rfmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = rfmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)

#varying max_depth
accuracies = np.zeros((2,20), float)
for k in range(0, 20):
    rfmodel = RandomForestClassifier(random_state=0, max_depth=k+1)
    rfmodel.fit(X_train, Y_train)
    Y_train_pred = rfmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[0,k] = acctr
    Y_test_pred = rfmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[1,k] = accte
plt.plot(range(1, 21), accuracies[0,:])
plt.plot(range(1, 21), accuracies[1,:])
plt.xlim(1,20)
plt.xticks(range(1, 21))
plt.xlabel('Max_depth')
plt.ylabel('Accuracy')
plt.title('Random Forest')
plt.show()

#varying n_estimators
accuracies = np.zeros((2,20), float)
ntrees = (np.arange(20)+1)*20
for k in range(0, 20):
    rfmodel = RandomForestClassifier(random_state=0, n_estimators=ntrees[k])
    rfmodel.fit(X_train, Y_train)
    Y_train_pred = rfmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[0,k] = acctr
    Y_test_pred = rfmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[1,k] = accte
plt.plot(ntrees, accuracies[0,:])
plt.plot(ntrees, accuracies[1,:])
plt.xticks(ntrees, rotation=90)
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.title('Random Forest')
plt.show()

#varying max_depth and n_estimators
mdepth = np.linspace(4, 8, 5)   
accuracies = np.zeros((4,5*20), float)
row = 0
for k in range(0, 5):
    for l in range(0, 20):
        rfmodel = RandomForestClassifier(random_state=0, max_depth=mdepth[k], n_estimators=ntrees[l])
        rfmodel.fit(X_train, Y_train)
        Y_train_pred = rfmodel.predict(X_train)
        acctr = accuracy_score(Y_train, Y_train_pred)
        accuracies[2,row] = acctr
        Y_test_pred = rfmodel.predict(X_test)
        accte = accuracy_score(Y_test, Y_test_pred)
        accuracies[3,row] = accte
        accuracies[0,row] = mdepth[k]
        accuracies[1,row] = ntrees[l]
        row = row + 1

#better visualization of the outcome
from tabulate import tabulate
headers = ["Max_Depth", "n_Estimators", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

#find highest test accuracy
print(accuracies[3].max())
maxi = np.array(np.where(accuracies==accuracies[3].max()))
print(maxi[0,:], maxi[1,:])
print(accuracies[:,maxi[1,:]])
table = tabulate(accuracies[:,maxi[1,:]].transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

#creation of a surface-plot to understand the relationship betweeen #
# the parameters and test accuracy #
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = accuracies[0,:]
y = accuracies[1,:]
z = accuracies[3,:]
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
ax.set_xlabel('Max_Depth')
ax.set_ylabel('n_Estimators')
ax.set_zlabel('accte')
plt.show()

#Model created using the optimized parameters
rfmodel = RandomForestClassifier(random_state=0, max_depth=8, n_estimators=60)
rfmodel.fit(X_train, Y_train)
Y_train_pred = rfmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = rfmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Random Forest', acctr, accte]

# View a list of the features and their importance scores
list(zip(X_train, rfmodel.feature_importances_))

################################
# Gradient Boosting Classifier #
################################

#model built 
from sklearn.ensemble import GradientBoostingClassifier
gbmodel = GradientBoostingClassifier(random_state=0)
gbmodel.fit(X_train, Y_train)

#model used on training and test data
Y_train_pred = gbmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = gbmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)

#varying max_depth(Boosting parameter)
accuracies = np.zeros((2,10), float)
for k in range(0, 10):
    gbmodel = GradientBoostingClassifier(random_state=0, max_depth=k+1)
    gbmodel.fit(X_train, Y_train)
    Y_train_pred = gbmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[0,k] = acctr
    Y_test_pred = gbmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[1,k] = accte
plt.plot(range(1, 11), accuracies[0,:])
plt.plot(range(1, 11), accuracies[1,:])
plt.xlim(1,10)
plt.xticks(range(1, 11))
plt.xlabel('Max_depth')
plt.ylabel('Accuracy')
plt.title('Gradient Boosting')
plt.show()

#varying learning_rate(Tree-based parameter)
accuracies = np.zeros((3,21), float)
lr = np.linspace(0, 0.4, 21)
lr[0] = 0.01
for k in range(0, 21):
    gbmodel = GradientBoostingClassifier(random_state=0, learning_rate=lr[k])
    gbmodel.fit(X_train, Y_train)
    Y_train_pred = gbmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[1,k] = acctr
    Y_test_pred = gbmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[2,k] = accte
    accuracies[0,k] = lr[k]
plt.plot(lr, accuracies[1,:])
plt.plot(lr, accuracies[2,:])
plt.xlabel('Learning_rate')
plt.ylabel('Accuracy')
plt.title('Gradient Boosting')
plt.show()

#varying max_depth and learning_rate
accuracies = np.zeros((4,21*10), float)
lr = np.linspace(0, 0.4, 21)
lr[0] = 0.01
row = 0
for k in range(0, 10):
    for l in range(0, 21):
        gbmodel = GradientBoostingClassifier(random_state=0, max_depth=k+1, learning_rate=lr[l])
        gbmodel.fit(X_train, Y_train)
        Y_train_pred = gbmodel.predict(X_train)
        acctr = accuracy_score(Y_train, Y_train_pred)
        accuracies[2,row] = acctr
        Y_test_pred = gbmodel.predict(X_test)
        accte = accuracy_score(Y_test, Y_test_pred)
        accuracies[3,row] = accte
        accuracies[0,row] = k+1
        accuracies[1,row] = lr[l]
        row = row + 1

#better visualization
from tabulate import tabulate
headers = ["Max_depth", "Learning_rate", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

#search for higher test accuracy
maxi = np.array(np.where(accuracies==accuracies[3].max()))
print(maxi[1,:])
print(accuracies[:,maxi[1,:]])
table = tabulate(accuracies[:,maxi[1,:]].transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

#create surface-plot for observing relationships better 
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = accuracies[0,:]
y = accuracies[1,:]
z = accuracies[3,:]
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
ax.set_xlabel('Max_depth')
ax.set_ylabel('Learning_rate')
ax.set_zlabel('accte')
plt.show()

#create a model with the optimized parameters
from sklearn.ensemble import GradientBoostingClassifier
gbmodel = GradientBoostingClassifier(random_state=0, max_depth=4, learning_rate=0.2)
gbmodel.fit(X_train, Y_train)
Y_train_pred = gbmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = gbmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Gradient Boosting', acctr, accte]


#########################
# Discriminant Analysis #
#########################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
dismodel = LinearDiscriminantAnalysis()
dismodel.fit(X_train, Y_train)
Y_train_pred = dismodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = dismodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Linear Discriminant Analysis', acctr, accte]

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qdismodel = QuadraticDiscriminantAnalysis()
qdismodel.fit(X_train, Y_train)
Y_train_pred = qdismodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = qdismodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Quadratic Discriminant Analysis', acctr, accte]


##################
# Neural Network #
##################

from sklearn.neural_network import MLPClassifier
nnetmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(17,), random_state=0)
nnetmodel.fit(X_train, Y_train)
Y_train_pred = nnetmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = nnetmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)

#tune the model by varying hidden layers
accuracies = np.zeros((3,20), float)
for k in range(0, 20):
    nnetmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(k+1,), random_state=0)
    nnetmodel.fit(X_train, Y_train)
    Y_train_pred = nnetmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[1,k] = acctr
    Y_test_pred = nnetmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[2,k] = accte
    accuracies[0,k] = k+1
plt.plot(range(1, 21), accuracies[1,:])
plt.plot(range(1, 21), accuracies[2,:])
plt.xlim(1,20)
plt.xticks(range(1, 21))
plt.xlabel('Hidden Neurons')
plt.ylabel('Accuracy')
plt.title('Neural Network')
plt.show()

#obtain the accuracies array
from tabulate import tabulate
headers = ["Hidden Neurons", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

#find maximal test accuracies
maxi = np.array(np.where(accuracies==accuracies[2:].max()))
table = tabulate(accuracies[:,maxi[1,:]].transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

#optimizing the model by using hidden layer with maximal accuracy
nnetmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(9,), random_state=0)
nnetmodel.fit(X_train, Y_train)
Y_train_pred = nnetmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = nnetmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Neural Network', acctr, accte]


##########################
# Support Vector Machine #
###########################

#linear kernel
from sklearn.svm import SVC
LinSVCmodel = SVC(kernel='linear', C=10, random_state=0)
LinSVCmodel.fit(X_train, Y_train)
Y_train_pred = LinSVCmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = LinSVCmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['SVM (Linear)', acctr, accte]

#Tuning the cost paramter
accuracies = np.zeros((3,21), float)
costs = np.linspace(0, 40, 21)
costs[0] = 0.5
for k in range(0, 21):
    LinSVCmodel = SVC(kernel='linear', C=costs[k], random_state=0)
    LinSVCmodel.fit(X_train, Y_train)
    Y_train_pred = LinSVCmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[1,k] = acctr
    Y_test_pred = LinSVCmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[2,k] = accte
    accuracies[0,k] = costs[k]
plt.plot(costs, accuracies[1,:])
plt.plot(costs, accuracies[2,:])
plt.xlim(1,20)
plt.xticks(costs, rotation=90)
plt.xlabel('Cost')
plt.ylabel('Accuracy')
plt.title('Linear SVM')
plt.show()

from tabulate import tabulate
headers = ["Cost", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

#radial kernel
from sklearn.svm import SVC
accuracies = np.zeros((3,21), float)
costs = np.linspace(0, 40, 21)
costs[0] = 0.5
for k in range(0, 21):
    RbfSVCmodel = SVC(kernel='rbf', C=costs[k], gamma=0.2, random_state=0)
    RbfSVCmodel.fit(X_train, Y_train)
    Y_train_pred = RbfSVCmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[1,k] = acctr
    Y_test_pred = RbfSVCmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[2,k] = accte
    accuracies[0,k] = costs[k]
plt.plot(costs, accuracies[1,:])
plt.plot(costs, accuracies[2,:])
plt.xlim(1,20)
plt.xticks(costs, rotation=90)
plt.xlabel('Cost')
plt.ylabel('Accuracy')
plt.title('Radial SVM')
plt.show()

from tabulate import tabulate
headers = ["Cost", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

#tuning cost parameter for radial kernel
accuracies = np.zeros((3,21), float)
gammas = np.linspace(0, 4.0, 21)
gammas[0] = 0.1
for k in range(0, 21):
    RbfSVCmodel = SVC(kernel='rbf', C=1, gamma=gammas[k], random_state=0)
    RbfSVCmodel.fit(X_train, Y_train)
    Y_train_pred = RbfSVCmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[1,k] = acctr
    Y_test_pred = RbfSVCmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[2,k] = accte
    accuracies[0,k] = gammas[k]
plt.plot(gammas, accuracies[1,:])
plt.plot(gammas, accuracies[2,:])
plt.xticks(gammas, rotation=90)
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.title('Radial SVM')
plt.show()

from tabulate import tabulate
headers = ["Gamma", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

n = 21
accuracies = np.zeros((4,n*n), float)
costs = np.linspace(0, 20, n)
costs[0] = 0.5
gammas = np.linspace(0, 4.0, n)
gammas[0] = 0.1
row = 0
for k in range(0, n):
    for l in range(0, n):
        RbfSVCmodel = SVC(kernel='rbf', C=costs[k], gamma=gammas[l], random_state=0)
        RbfSVCmodel.fit(X_train, Y_train)
        Y_train_pred = RbfSVCmodel.predict(X_train)
        acctr = accuracy_score(Y_train, Y_train_pred)
        accuracies[2,row] = acctr
        Y_test_pred = RbfSVCmodel.predict(X_test)
        accte = accuracy_score(Y_test, Y_test_pred)
        accuracies[3,row] = accte
        accuracies[0,row] = costs[k]
        accuracies[1,row] = gammas[l]
        row = row + 1

from tabulate import tabulate
headers = ["Cost", "Gamma", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

maxi = np.array(np.where(accuracies==accuracies[3].max()))
print(maxi[1,:])
print(accuracies[:,maxi[1,:]])
table = tabulate(accuracies[:,maxi[1,:]].transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = accuracies[0,:]
y = accuracies[1,:]
z = accuracies[3,:]
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
ax.set_xlabel('Cost')
ax.set_ylabel('Gamma')
ax.set_zlabel('accte')
plt.show()

RbfSVCmodel = SVC(kernel='rbf', C=2, gamma=2.6 , random_state=0)
RbfSVCmodel.fit(X_train, Y_train)
Y_train_pred = RbfSVCmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = RbfSVCmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['SVM (Radial)', acctr, accte]


#polynomial kernel
n = 21
accuracies = np.zeros((4,n*n), float)
costs = np.linspace(0, 20, n)
costs[0] = 0.5
degrees = np.linspace(0, 10.0, n)
degrees[0] = 0.1
row = 0
for k in range(0, n):
    for l in range(0, n):
        PolySVCmodel = SVC(kernel='poly', C=costs[k], degree=degrees[l], random_state=0)
        PolySVCmodel.fit(X_train, Y_train)
        Y_train_pred = PolySVCmodel.predict(X_train)
        acctr = accuracy_score(Y_train, Y_train_pred)
        accuracies[2,row] = acctr
        Y_test_pred = PolySVCmodel.predict(X_test)
        accte = accuracy_score(Y_test, Y_test_pred)
        accuracies[3,row] = accte
        accuracies[0,row] = costs[k]
        accuracies[1,row] = gammas[l]
        row = row + 1

from tabulate import tabulate
headers = ["Cost", "Degree", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

maxi = np.array(np.where(accuracies==accuracies[3:].max()))
print(maxi[1,:])
print(accuracies[:,maxi[1,:]])
table = tabulate(accuracies[:,maxi[1,:]].transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)


################
# Final Report #
################

print(report)




