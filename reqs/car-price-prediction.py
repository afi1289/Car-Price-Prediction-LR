#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as po
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import random
from plotly.offline import iplot
pd.options.plotting.backend= "plotly"

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Exploratory data analysis:
# One of the most important steps in ML and Data Science project is understanding data. At this step we are going to:
# * Examine the features 
# * Check features distribution
# * Examine the dependent variable (The variable we are trying to predict)
# * Assess the data quality
# * Check the missing values
# 
# In our data analysis toolbox, we will use **plotly** for visualization. **Plotly** is an interactive framework that allows users to plot all Matplotlib and Seaborn charts with a full control over what is being plotted.

# In[3]:


# Load the Dataset into a DataFrame
# NOTE: in practice this step can be more than one line of code
df = pd.read_csv('../input/cardataset/data.csv')


# In[4]:


# Check the length of the dataframe
len(df)


# In[5]:


# Check the data types 
df.dtypes


# In[6]:


# Show the first 10 rows to get an idea how the data look like
df.head(10)


# * We have almost 12k cars in our dataset!
# * There are some incosistencies: 
#     * Column names sometimes have spaces and underscores (e.g., `Engine HP` and `Driven_Wheels`)
#     * Some featre values are capitalized (e.g., `Transmission type`) or they are just strings with spaces
#     
#     Therefore we have to make some **preprocessing** steps:  

# In[9]:


# Let's lowercase all the column names and replace spaces with _
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns


# In[10]:


# Select only columns with string values (i.e., we are not going to preprocess columns with numeric values)
string_columns= list(df.dtypes[df.dtypes == 'object'].index)
string_columns


# In[16]:


# Let's lowercase and replace spaces with underscores for values in all string columns of the df
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')


# In[18]:


# Check the data again
df.head()


# Our dataframe looks better now! But what the columns actually mean?
# * `make`: maker of a car (BMW, Toyota, and so on)
# * `model`: model of car
# * `year`: year when the car was manufactured
# * `engine_feul_type`: type of fuel the engine needs
# * `engine_hp`: horsepower of the engine
# * `engine_cylinders`: number of cylinders in the engine
# * `transmission_type`: automatic or manual
# * `driven_wheels`: fromn, rear, all
# * `number_of_doors`
# * `market_category`: luxury, crossover, and so on
# * `vehicle_style`: sedan or convertible
# * `highway_mpg`: miles per gallon on the highway
# * `city_mpg`: miles per gallon n the city
# * `popularity`: number of times the car was mentioned in a Twitter stream
# 
# and the most important column is the price of the car:
# * `msrp`: manufacturer's suggested retail price. This column is our **target variable** which we are trying to predict!

# ## Target Variable Analysis

# ### What the values of **Target variable** look like? Visualize them using a **Histogram**:

# In[19]:


fig = go.Figure()
fig.add_trace(go.Histogram (x=df['msrp'],
                            marker=dict(color='rgba(171, 50, 96,0.6)')
                            
                            )
             )
iplot(fig)


# ### Long tail? **log transformation**!
# The long tail would confuse the model. In other words, the model will NOT learn enough.  To solve this Problem we can use log transformation later to get the following result:

# In[20]:


log_price = np.log1p(df['msrp'])


# In[21]:


fig = go.Figure()
fig.add_trace(go.Histogram (x=log_price,
                            marker=dict(color='rgba(171, 50, 96,0.6)')
                            
                            )
             )
iplot(fig)


# ### Normal or like Normal? Linear Regression works!
# * If the target variable (before/after log transformation) has a normal distribution or normal-like distriubtion, the linear regression model can perform well.
# * In our case, the distribution of the target variable is not normal because of the large peak in lower prices, but the model can deal with it more easily.

# ## Something is Missing? Check for **missing values**:
# 

# ML algorithms cannot deal with missing values automatically. We need to **deal with them** before modeling step. Let's look at my Data Cleaning workflow

# ![Data Cleaning (3).png](attachment:712142f5-0066-48f0-af60-6826b078df76.png)

# In[ ]:


# check missing values for each column
df.isnull().sum()


# ## Validation Framwork

# We split the dataset in order to ensure that our model can generalize on new/unseen data. Therefore we're going to use two methods.

# ### 1. Splitting Data into validation, test, and training sets

# In[ ]:


from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)
df_train, df_val = train_test_split(df_train, test_size = 0.2, random_state = 42)


# In[ ]:


df_train.head()


# ## Normlize the target variable values
# Our target variable (car prices `msrp`) has a long tail in the distribution of its values. We have to remove this effect by applying log transformation. We will write functions to do that, since it allow u to reproduce these transformations easily. In addition, this enables us to try various transformations and see which combination of transformations works best.

# In[ ]:


# Return the natural logarithm of one plus the input array, element-wise
# +1 is important part in cases that have zeros
def transform_func(df, target):
    y_norm = np.log1p(df[target].values)
    return y_norm

# undo the logarithm and apply the exponent function (for final prediction)
def inverse_func(y_norm):
    return np.expm1(y_norm)


# In[ ]:


# apply the transformation
y_train = transform_func(df_train, "msrp")
y_val = transform_func(df_val, "msrp")
y_test = transform_func(df_test, "msrp")


# In[ ]:


# now the values of the target variable are normalized
fig = go.Figure()
fig.add_trace(go.Histogram (x=y_train, 
                            marker=dict(color='#636EFA'),
                            name='Training'
                            
                            )
             )
fig.add_trace(go.Histogram (x=y_val, 
                            marker=dict(color='#EF553b'),
                            name='Validation'
                            
                            )
             )
fig.add_trace(go.Histogram (x=y_test, 
                            marker=dict(color='#00CC96'),
                            name='Testing'
                            
                            )
             )
iplot(fig)


# Training points are more frequent than the others. Therefore, the head of training distribution is bigger.

# In[ ]:


# remove the target variable from the dataframe to avoid using it in training later
del df_train['msrp']
del df_val['msrp']
#del df_test['msrp']


# ## Implement Linear Regression
# 

# ### Base Solution

# In this basic solution, we are going to implement the **normal equation** that will find the **optimal** weights and biases:
# 
# $ w = \left ( X^{T}X \right )^{-1}X^{T}y $
# 
# Therefore, we need create a **matrix of features** $ X $ from our dataframe. We will select the following features:
# * `engine_hp`
# * `engine_cylinders`
# * `highway_mpg`
# * `city_mpg`
# * `popularity`

# In[ ]:


# implement the normal equation to find the optimal weights and biases

def train_linear_regression(X, y):
    # create an array that contains only ones (it is the dummy feature)
    ones = np.ones(X.shape[0])
    # add the array of ones as the first column of X
    X = np.column_stack([ones, X])
    
    # normal equation formula
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    
    w = XTX_inv.dot(X.T).dot(y)
    
    # return the optimal biases and weights
    return w[0],w[1:]


# In[ ]:


# select features and write them to  new variable
base = ['engine_hp','engine_cylinders','highway_mpg','city_mpg','popularity']
df_num = df_train[base]


# As mentioned before, the dataset has missing values. Unfortunately, ML methods cannot deal with them. Therefore, we are going to fill missing values with `0`:

# In[ ]:


# fill the missing values
df_num = df_num.fillna(0)


# In this step, we convert our dataframe to a NumPy array

# In[ ]:


# convert the dataframe to numpy array
X_train = df_num.values


# In[ ]:


# use the normal equation function to get the optimal biases and weights
w_0, w = train_linear_regression(X_train, y_train)


# In[ ]:


# apply the new biases and weights on the training data to see how well it predicts (theoretical step)
y_pred = w_0 + X_train.dot(w)


# In[ ]:


# visualize to see how good the predictions are
# compare the distribution of traget variable with the distribution of predicted values
fig = go.Figure()
fig.add_trace(go.Histogram (x=y_train, 
                            marker=dict(color='#636EFA'),
                            name='Actual'
                            
                            )
             )
fig.add_trace(go.Histogram (x=y_pred, 
                            marker=dict(color='#EF553b'),
                            name='prediction'
                            
                            )
             )
fig.update_layout(                       
                    title = 'Actual Values vs. Predicted Values', # Title
                    )

iplot(fig)


# The result above indicates that our model is **not powerful** enough to capture the distribution of the **target variable**. However, comparing distributions is not always feasible. Let's evaluate model performance with a metric such as **root mean squared error RMSE**

# In[ ]:


# RMSE implementaion
def rmse(y,y_pred):
    # computes the difference between the actual values and predictions
    error = y_pred - y
    # compute the squared error, and then calculates its mean
    mse = (error**2).mean()
    
    #return the root
    return np.sqrt(mse)


# In[ ]:


rmse(y_train, y_pred)


# This number tells us that on average, the model's predictions are off by 0.75. **NOTE** This result alone my not be very useful, but we can use it to compare this model with other models.

# ### Validating the model

# To validate the model, we will use the validation dataset. Therefore, we will create `X_val` matrix follwoing the same steps for `X_train` using a function

# In[ ]:


# prepare function to convert a Dataframe into matrix
def prepare_X(df):
    df_num = df[base]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X


# In[ ]:


X_val = prepare_X(df_val)


# In[ ]:


# apply the model (i.e., weights and biases) on X_val to get predictions
y_pred = w_0 + X_val.dot(w)


# In[ ]:


# evaluate the model performance using RMSE
rmse(y_val, y_pred)


# This number should be used to compare models. Before that, let's do some feature engineering

# ## Simple Feature Engineering:
# To improve our model, we cann add more features to the model: we create others and add them to the existing features. This process called **feature engineering**.

# 1. Create a new feature, `age` from the feature `year`

# In[ ]:


# the dataset was created in 2017
#df_train['age'] = 2017 - df_train['year']


# If you have to repeat the same feature extraction step multiple times, put its logic into preprocessing function: let's put the previous logic into the `prepare_X` function

# In[ ]:


# createing the age feature in the prepare_X function
def prepare_X(df):
    #create a copy of the input parameter to prevent side effects
    df = df.copy()
    # create a copy of the base list with basic features
    features = base.copy()
    
    # compute the age feature
    df['age'] = 2017 - df['year']
    # append age to the list of feature used for the model
    features.append('age')
    
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X


# In[ ]:


# test if adding the feature age leads to any improvements
X_train = prepare_X(df_train)
w_0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)

print('validation: ', rmse(y_val, y_pred))


# The validation error is **improved**! It was 0.76 and it is now 0.517. Let's look at the distribution of redicted values:

# In[ ]:


# visualize to see how good the predictions are
# compare the distribution of traget variable with the distribution of predicted values
fig = go.Figure()
fig.add_trace(go.Histogram (x=y_val, 
                            marker=dict(color='#636EFA'),
                            name='Actual'
                            
                            )
             )
fig.add_trace(go.Histogram (x=y_pred, 
                            marker=dict(color='#EF553b'),
                            name='Prediction'
                            
                            )
             )
fig.update_layout(                       
                    title = 'Actual Values vs. Predicted Values', # Title
                    )

iplot(fig)


# The **distribution of the predictions** follows the **target distribution** a lot more closely than previously!

# ## Handling Categorical Variables

# You can use categorical variables in a ML model in multiple ways. One of the simplest eays is to use encode such variables by a set of binary features, with a separate feature for each distinct value (**one-hot encoding**)

# In[ ]:


# simple one-hot encoding for the feature 'number_of_doors'
features = base.copy()


# iterat over possible values of the 'number_of_doors' variable
for v in [2,3,4]: 
    # give a feature a meaningful name, such as "num_doors_2" for v=2
    feature = "number_doors_%s" % v
    # create the one-hot encoding feature
    value = (df['number_of_doors'] == v).astype(int)
    
    # add the feature back to the dataframe
    #df[feature]= value # uncomment to understand the logic
    features.append(feature)
    


# In[ ]:


# handling categorical variables in the prepare_X function

def prepare_X(df):
    df = df.copy()
    features = base.copy()

    df['age'] = 2017 - df['year']
    features.append('age')
    
    # encode "number_of_doors" feature
    for v in [2, 3, 4]:
        feature = 'num_doors_%s' % v
        df[feature] = (df['number_of_doors'] == v).astype(int)
        features.append(feature)
        
    # encode "make" feature
    for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
        feature = 'is_make_%s' % v
        df[feature] = (df['make'] == v).astype(int)
        features.append(feature)
        
    # encode the "engine_fuel_type" feature
    for v in ['regular_unleaded', 'premium_unleaded_(required)', 
              'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)']:
        feature = 'is_type_%s' % v
        df[feature] = (df['engine_fuel_type'] == v).astype(int)
        features.append(feature)
    
    # encode the "transmission_type" feature
    for v in ['automatic', 'manual', 'automated_manual']:
        feature = 'is_transmission_%s' % v
        df[feature] = (df['transmission_type'] == v).astype(int)
        features.append(feature)
    
    # encode the "driven_wheels" feature
    for v in ['front_wheel_drive', 'rear_wheel_drive', 'all_wheel_drive', 'four_wheel_drive']:
        feature = 'is_driven_wheens_%s' % v
        df[feature] = (df['driven_wheels'] == v).astype(int)
        features.append(feature)

    # encode the "market_category" feature
    for v in ['crossover', 'flex_fuel', 'luxury', 'luxury,performance', 'hatchback']:
        feature = 'is_mc_%s' % v
        df[feature] = (df['market_category'] == v).astype(int)
        features.append(feature)

    # encode the "vehicle_size" feature
    for v in ['compact', 'midsize', 'large']:
        feature = 'is_size_%s' % v
        df[feature] = (df['vehicle_size'] == v).astype(int)
        features.append(feature)

    # encode the "vehicle_style" feature
    for v in ['sedan', '4dr_suv', 'coupe', 'convertible', '4dr_hatchback']:
        feature = 'is_style_%s' % v
        df[feature] = (df['vehicle_style'] == v).astype(int)
        features.append(feature)

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X


# In[ ]:


# test if adding the feature age leads to any improvements
X_train = prepare_X(df_train)
w_0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)

print('validation: ', rmse(y_val, y_pred))


# The result is significantly worse than before. The new features made the score a lot **worse**. The reason for this behavior is numerical instability (normal equation solution).numerical instability issues are typically solved with **regularization**:

# In[ ]:


# add regularization to the training function
def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]


# In[ ]:


X_train = prepare_X(df_train)
w_0, w = train_linear_regression_reg(X_train, y_train, r=0.01)

y_pred = w_0 + X_train.dot(w)
print('train', rmse(y_train, y_pred))

X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)
print('val', rmse(y_val, y_pred))


# This result is an **improvement** over the previous score.

# ### Using the model:

# We can use our model to predict the price of a specific car. Suppose that we want to predict the price of the following car:

# In[ ]:


example = df_test.iloc[2].to_dict()
example 


# In[ ]:


# create a dataframe with one row
df_example = pd.DataFrame([example])  # https://stackoverflow.com/questions/17839973/constructing-pandas-dataframe-from-values-in-variables-gives-valueerror-if-usi
df_example


# In[ ]:


# convert the dataframe to numpy array
X_example = prepare_X(df_example)#[0]
X_example


# In[ ]:


# apply the new biases and weights on the example to get a prediction 
y_pred = w_0 + X_example.dot(w)
y_pred


# However, y_pred is NOT the final prediction! it is the **logarithm** of the price. Therefore, we have to undo the logarithm and apply the **exponent function**:

# In[ ]:


final_prediction = inverse_func(y_pred)
final_prediction = np.expm1(y_pred)
final_prediction


# The **final prediction** is 39,478  The **real price** is 37,650 So our model is not far from the actual price.

# # Next steps

# * create a numerical transformation pipelene
# `from sklearn.pipeline import Pipeline # handle numerical`
# 
# * create a categorical transformation pipeline
# `from sklearn.preprocessing import OneHotEncoder`
# 
# * create a single transformer that can handle numerical and categorical variables p.74 hands-on
# `from sklearn.compose import ColumnTransformer`
# 
# * It is almost always preferable to have at least a little bit of regularization, so generally you should avoid plain linear Regression. Ridge is a good default.

# In[ ]:


### simple imputer start ###
# Handling missing values using sklearn imputer (Hands-on Machine learning p.67)
from sklearn.impute import SimpleImputer

imputer = SimpleInputer(missing_values = 0, strategy="mean")


# In[ ]:


# understand the logic of simple imputer based on documentation

from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=0, strategy='mean')
imp_mean.fit([[7, 2, 3], [4, 0, 6], [10, 5, 9]])

X = [[0, 2, 3], [4, 0, 6], [10, 0, 9]]
print(imp_mean.transform(X))

