#!/usr/bin/env python
# coding: utf-8

# # Group Members:
# ## Dewansh Singh (037)
# ## Sakshi Dumbre (026)
# ## Anurag Taparia (068)
# ## Vedant Dawange (082)

# ## Importing Libraries

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# # Step-1: Gathering Data

# In[2]:


df=pd.read_csv("student2.csv")
df.head()


# In[3]:


# let's preview the dataset

df.head(15)


# In[4]:


# view dimensions of dataset

df.shape


# ## Statistical Method

# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


df.info()


# # Exploratory Data Analysis (EDA)
# # Step-2: Data Preparation
# ## Data Cleaning

# In[8]:


#invalid values
df.loc[(df['Status'] != 'Active') & (df['Status'] != 'Inactive')]


# In[9]:


df= df.drop([5])
df= df.drop([41])
df.reset_index()


# In[10]:


df['Graduation Year'].fillna((df['Graduation Year'].mean()), inplace=True)


# In[11]:


df.isnull().sum()


# # Step-3: Data Wrangling

# In[12]:


df.head(15)


# In[13]:


df=pd.read_csv("student2.csv")


# In[14]:


df.shape


# In[15]:


#invalid values
df.loc[(df['Status'] != 'Active') & (df['Status'] != 'Inactive')]


# In[16]:


df= df.drop([5])
df= df.drop([41])
df.reset_index()


# In[17]:


# view dimensions of dataset after droping the columns having invalid values

df.shape


# In[18]:


df = df.dropna()
df.head(15)


# In[19]:


df.groupby('Status').mean()


# In[20]:


df.isnull().sum()


# In[21]:


df.Status.replace(('Active', 'Inactive'), (1, 0), inplace=True)
df.head(10)


# # Step-4: Analyse Data
# ## Data Visualization using several Graphical methods 

# In[22]:


df.hist(figsize=(10,8))


# In[23]:


sns.heatmap(df.corr(),annot=True)


# In[24]:


sns.pairplot(df, height=2.5)


# In[25]:


fig, axes = plt.subplots(nrows=1,ncols=1)
fig.set_size_inches(5, 5)
sns.boxplot(data=df,y="Status",x="Graduation Year", palette="Oranges")


# In[26]:


labels = df.index
colors = ['lightskyblue', 'red']
plt.pie(df['Status'], labels= labels, colors=colors, startangle=2, autopct='%.1f%%')
plt.show()


# **Now, as we can observe there is no Outlier's available so now our dataset is fully cleaned. Ready to Train the model using this dataset.**

# # Step-5: Train Model

# In[28]:


col=['Quiz Duration(in min)','Grade','Tab Switch Log']
x=df[col]
y=df['Status']


# In[29]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)


# In[30]:


x_train.shape


# In[31]:


x_test.shape


# ## Decision Tree
# 
# A decision tree is a **flowchart-like tree structure** where an internal node represents feature(or attribute), the branch represents a decision rule, and each leaf node represents the outcome. The topmost node in a decision tree is known as the root node. It learns to partition on the basis of the attribute value.  It partitions the tree in a recursive manner called recursive partitioning. **This flowchart-like structure helps you in decision making**. It can be visualized like a flowchart diagram which easily mimics human level thinking. That is why decision trees are easy to understand and interpret.
# 

# 
# 
# Another decision tree algorithm CART (Classification and Regression Tree) uses the Gini method to create split points.
# 
# $$Gini(D) = 1 - \sum_{i = 1}^{m} P_i^2 $$

# ### Building Decision Tree Model
# 
# Let's create a Decision Tree Model using Scikit-learn.

# In[32]:


def feature_imp(model):
    importance = model.feature_importances_

    for i,v in enumerate(importance):
        imp[i]=v;
        print('Feature: %0d, Score: %.5f' % (i,imp[i]))

    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()


# In[33]:


from matplotlib import pyplot

clf_gini = DecisionTreeClassifier(criterion='gini',random_state=0)
clf_ginifit = clf_gini.fit(x_train,y_train)
y_pred_gini = clf_ginifit.predict(x_test)

feature_imp(clf_gini)


# ### Compute precision, recall, F-measure and support
# 
# The _accuracy_ is the ratio *tp + tn = (tp + tn + fp + fn)*<br> 
# 
# The _precision_ is the ratio *tp / (tp + fp)* where *tp is the number of true positives* and *fp the number of false positives*. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.<br>
# 
# The _recall_ is the ratio *tp / (tp + fn)* where *tp is the number of true positives* and *fn the number of false negatives*. The recall is intuitively the ability of the classifier to find all the positive samples.<br>
# 
# The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.<br>
# 
# The F-beta score weights recall more than precision by a factor of beta. beta == 1.0 means recall and precision are equally important.<br>
# 
# *The support is the number of occurrences of each class in y_test.*

# # Step - 6 Test Model

# In[34]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred_gini))
print("Precision:",metrics.precision_score(y_test, y_pred_gini))
print("Recall:",metrics.recall_score(y_test, y_pred_gini))
print("F1 Score:",metrics.f1_score(y_test, y_pred_gini))


# **Well, we got a classification rate of 96.66%, considered as best accuracy. So no need of improvement and this shows that the dataset we got is perfectly fine. No need of pruning is required. But we will still look for some pruning methods so that still we can improve our accuracy.**

# In[35]:


fig = plt.figure(figsize=(100,50))
DT_gini = tree.plot_tree(clf_gini, 
                   feature_names=col,  
                   class_names=['0','1'],
                   filled=True)


# ## Optimizing Decision Tree Performance
# 
# - **criterion :  optional (default=”gini”) or Choose attribute selection measure**: This parameter allows us to use the different-different attribute selection measure. Supported criteria are “gini” for the Gini index and “entropy” for the information gain.
# 
# 
# - **splitter : string, optional (default=”best”) or Split Strategy**: This parameter allows us to choose the split strategy. Supported strategies are “best” to choose the best split and “random” to choose the best random split.
# 
# 
# - **max_depth : int or None, optional (default=None) or Maximum Depth of a Tree**: The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples. The higher value of maximum depth causes overfitting, and a lower value causes underfitting.
# 
# In Scikit-learn, optimization of decision tree classifier performed by only pre-pruning. Maximum depth of the tree can be used as a control variable for pre-pruning. 

# ### Gini

# In[36]:


clf_gini4 = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=0)
clf_fitgini4 = clf_gini4.fit(x_train,y_train)
y_pred_gini4 = clf_fitgini4.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_gini4))


# In[37]:


fig = plt.figure(figsize=(25,10))
DT_gini4 = tree.plot_tree(clf_gini4, 
                   feature_names=col,  
                   class_names=['0','1'],
                   filled=True)


# ## Comparison of Decision Tree at different *max_depth*.

# In[38]:


clf_gini3 = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
clf_fitgini3 = clf_gini3.fit(x_train,y_train)
y_pred_gini3 = clf_fitgini3.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_gini3))


# In[39]:


fig = plt.figure(figsize=(15,5))
DT_gini3 = tree.plot_tree(clf_gini3, 
                   feature_names=col,  
                   class_names=['0','1'],
                   filled=True)


# In[40]:


clf_gini2 = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=0)
clf_fitgini2 = clf_gini2.fit(x_train,y_train)
y_pred_gini2 = clf_fitgini2.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_gini2))


# In[41]:


fig = plt.figure(figsize=(7,5))
DT_gini2 = tree.plot_tree(clf_gini2, 
                   feature_names=col,  
                   class_names=['0','1'],
                   filled=True)


# ## Entropy

# We can plot a decision tree on the same data with max_depth=3.  Other than pre-pruning parameters like Gini used initially, now we can also try other attribute selection measure such as <b>entropy</b>.
# 

# In[42]:


clf_entropy = DecisionTreeClassifier(criterion='entropy',random_state=0)
clf_entropyfit = clf_entropy.fit(x_train,y_train)
y_pred_entropy = clf_entropyfit.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_entropy))
print("Precision:",metrics.precision_score(y_test, y_pred_entropy))
print("Recall:",metrics.recall_score(y_test, y_pred_entropy))
print("F1 Score:",metrics.f1_score(y_test, y_pred_entropy))


# In[43]:


fig = plt.figure(figsize=(100,50))
DT_entropy = tree.plot_tree(clf_entropy, 
                   feature_names=col,  
                   class_names=['0','1'],
                   filled=True)


# In[44]:


clf_entropy4 = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0)
clf_fitentropy4 = clf_entropy4.fit(x_train,y_train)
y_pred_entropy4 = clf_fitentropy4.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_entropy4))


# In[114]:


fig = plt.figure(figsize=(10,5))
DT_entropy4 = tree.plot_tree(clf_entropy4, 
                   feature_names=col,  
                   class_names=['0','1'],
                   filled=True)


# In[115]:


clf_entropy3 = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf_fitentropy3 = clf_entropy3.fit(x_train,y_train)
y_pred_entropy3 = clf_fitentropy3.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_entropy3))


# In[116]:


fig = plt.figure(figsize=(10,5))
DT_entropy3 = tree.plot_tree(clf_entropy3, 
                   feature_names=col,  
                   class_names=['0','1'],
                   filled=True)


# In[117]:


clf_entropy2 = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=0)
clf_fitentropy2 = clf_entropy2.fit(x_train,y_train)
y_pred_entropy2 = clf_fitentropy2.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_entropy2))


# **As we can observe that with all possible criteria we have tried the possibilities of improving the accuracy but we are getting same accuracy this shows that without pruning we can get the best accuracy and this concludes that our dataset is perfectly refined**

# Measure and test error to measure how many mistakes this tree has made on test data respectively.

# In[118]:


fig = plt.figure(figsize=(7,5))
DT_entropy2 = tree.plot_tree(clf_entropy2, 
                   feature_names=col,  
                   class_names=['0','1'],
                   filled=True)


# ### % of Error made by model

# In[119]:


from sklearn.metrics import mean_squared_error
test_error = mean_squared_error(y_test,y_pred_gini)
print(test_error)


# **Hence, our model is giving 96% accuracy for the collected dataset and we can conclude that with 96% of accuracy our model best fit ofr this algorithm and the attentiveness and activeness can be concluded with this model**

# In[45]:


import matplotlib.pyplot as plt

 
active = [1]
inactive = [0]
slices = [70,30]
activities = ['active','inactive']
cols = ['c','m']
 
plt.pie(slices,
  labels=activities,
  colors=cols,
  startangle=90,
  shadow= True,
  explode=(0,0.1),
  autopct='%1.1f%%')
 
plt.title('Pie Plot representation of Status(Result)')
plt.show()


# In[ ]:




