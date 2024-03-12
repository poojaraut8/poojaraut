#!/usr/bin/env python
# coding: utf-8

# #                                           # Crop Yield Prediction

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[2]:


df=pd.read_csv("yield_df.csv") 
df.head()


# In[3]:


df.tail()


# In[4]:


df.drop('Unnamed: 0',axis=1,inplace=True)


# In[5]:


df.head()


# In[6]:


df.isnull().sum()


# In[7]:


df.shape


# In[8]:


df.duplicated().sum()


# In[9]:


df.drop_duplicates(inplace=True)


# In[10]:


df.duplicated().sum()


# In[12]:


df.describe()


# In[13]:


df.info()


# In[13]:


numerical_columns =df.select_dtypes(include=['int64', 'float64']).columns.tolist()


# In[14]:


correlation_matrix =df[numerical_columns].corr()


# In[15]:


correlation_matrix.corr()


# In[16]:


df['average_rain_fall_mm_per_year']


# # Checking Outliers

# In[17]:


plt.figure(figsize=(14,10))
sns.set_style(style='whitegrid')
plt.subplot(2,2,1)
sns.boxplot(x='hg/ha_yield',data=df)
plt.subplot(2,2,2)
sns.boxplot(x='average_rain_fall_mm_per_year',data=df)
plt.subplot(2,2,3)
sns.boxplot(x='pesticides_tonnes',data=df)
plt.subplot(2,2,4)
sns.boxplot(x='avg_temp',data=df)


# In[18]:


print("Highest allowed",df['hg/ha_yield'].mean()+3*df['hg/ha_yield'].std())
print("Lowest allowed",df['hg/ha_yield'].mean()-3*df['hg/ha_yield'].std())


# In[19]:


print("Highest allowed",df['pesticides_tonnes'].mean()+3*df['pesticides_tonnes'].std())
print("Lowest allowed",df['pesticides_tonnes'].mean()-3*df['pesticides_tonnes'].std())


# In[20]:


print("Highest allowed",df['avg_temp'].mean()+3*df['avg_temp'].std())
print("Lowest allowed",df['avg_temp'].mean()-3*df['avg_temp'].std())


# In[21]:


df[(df['hg/ha_yield']>156793.60438832606)|(df['hg/ha_yield']<-68077.52093896193)]


# In[22]:


df[(df['pesticides_tonnes']>96586.80209087113)|(df['pesticides_tonnes']<-51678.80822750837)]


# In[23]:


df[(df['avg_temp']>39.645814232383266)|(df['avg_temp']<1.3175061054983885)]


# In[24]:


new_df=df[(df['hg/ha_yield']<156793.60438832606)&(df['hg/ha_yield']>-68077.52093896193)]
new_df


# In[25]:


new_df=df[(df['pesticides_tonnes']<96586.80209087113)&(df['pesticides_tonnes']>-51678.80822750837)]
new_df


# In[26]:


new_df=df[(df['avg_temp']<39.645814232383266)&(df['avg_temp']>1.3175061054983885)]
new_df


# In[27]:


new_df=df.loc[(df['hg/ha_yield']<156793.60438832606)&(df['hg/ha_yield']>-68077.52093896193)]
print('Before removing outliers:',len(df))
print('After removing outliers:',len(new_df))
print('Outliers:',len(df)-len(new_df))


# In[28]:


new_df=df.loc[(df['pesticides_tonnes']<96586.80209087113)&(df['pesticides_tonnes']>-51678.80822750837)]
print('Before removing outliers:',len(df))
print('After removing outliers:',len(new_df))
print('Outliers:',len(df)-len(new_df))


# In[29]:


new_df=df.loc[(df['avg_temp']<39.645814232383266)&(df['avg_temp']>1.3175061054983885)]
print('Before removing outliers:',len(df))
print('After removing outliers:',len(new_df))
print('Outliers:',len(df)-len(new_df))


# In[30]:


new_df


# # Visualization

# # histplot for average rainfall per year vs frequency

# In[31]:


plt.figure(figsize=(10, 6))
sns.histplot(df['average_rain_fall_mm_per_year'], bins=20, kde=True)
plt.title('Histogram of Average Rainfall per Year')
plt.xlabel('Average Rainfall (mm per year)')
plt.ylabel('Frequency')
plt.show()


# # graph frequency vs area

# In[32]:


#graph frequency vs area

plt.figure(figsize=(10,20))
sns.countplot(y=df['Area'])


# # yield per country

# In[33]:


len(df['Area'])


# In[34]:


#yield per country
Country=(df['Area'].unique())


# In[35]:


for state in Country:
    print(state)


# In[36]:


yield_per_country = []
for state in Country:
    yield_per_country.append(df[df['Area']==state]['hg/ha_yield'].sum())


# In[37]:


df['hg/ha_yield'].sum()


# In[38]:


#yeild per country graph

plt.figure(figsize=(10,20))
sns.barplot(y=Country,x= yield_per_country)


# # Heatmap

# In[39]:


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix.corr(), annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()


# In[40]:


df['Item'].value_counts()


# # Count of items

# In[41]:


sns.countplot(y=df['Item'])


# In[42]:


#yeild vs item

crops = (df['Item'].unique())


# In[43]:


len(crops)


# In[44]:


yield_per_crop= []
for crop in crops:
    yield_per_crop.append(df[df['Item']==crop]['hg/ha_yield'].sum())


# In[45]:


yield_per_crop


# # Feature Engineering

# # train,test ans Split

# In[46]:



col=['Year','average_rain_fall_mm_per_year','pesticides_tonnes','avg_temp','Area','Item','hg/ha_yield']
df=df[col]


# In[47]:


df


# In[48]:


x=df.drop('hg/ha_yield',axis=1)
y=df['hg/ha_yield']


# In[49]:


from sklearn.model_selection import train_test_split


# In[50]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[51]:


x_train.shape


# In[52]:


x_test.shape


# In[53]:


x_train


# # converting categorical to numeric and scaling the values

# In[54]:


from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer


# In[55]:


ohe = OneHotEncoder(drop='first')
Scaler = StandardScaler()


# In[56]:


preprocesser = ColumnTransformer(
transformers=[
    ('OneHotEncoder',ohe,[4,5]),
    ('Standardization',Scaler,[0,1,2,3])
],
remainder='passthrough'
)


# In[57]:


preprocesser 


# In[58]:


x_train_dummy = preprocesser.fit_transform(x_train)
x_test_dummy = preprocesser.transform(x_test)


# In[59]:


x_train_dummy


# # Training model

# In[60]:


#training model
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# In[61]:


models = {
    'LinearRegression': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'DecisionTreeRegressor': DecisionTreeRegressor()
}

for name, mod in models.items():
    mod.fit(x_train_dummy, y_train)
    y_pred = mod.predict(x_test_dummy)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)  # Add MAE calculation
    rmse = np.sqrt(mse)  # Calculate RMSE from MSE
    
    r2 = r2_score(y_test, y_pred)
    
    print(f"{name}:\nMSE: {mse}\nMAE: {mae}\nRMSE: {rmse}\nR2 Score: {r2}\n")


# # Accuracy Graph 

# In[63]:


import matplotlib.pyplot as plt

# Data
names = list(models.keys())
mse_scores = [1821709883.1611226, 1822234158.2996287, 1822003458.8200076, 127461418.18759204, 168880944.36649317]
mae_scores = [31935.09868323844, 31943.050839987888, 31939.334054808297, 6907.133485586494, 7614.02278828204]
rmse_scores = [42680.15346440001, 42683.78406976899, 42681.24353502544, 11286.145601464146, 12999.329708695043]
r2_scores = [0.7486565577888948, 0.7485842229351424, 0.74861605281204, 0.9824139991265859, 0.9766992986790629]

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Line chart for MSE
color = 'tab:blue'
ax1.set_xlabel('Models', fontweight='bold')
ax1.set_ylabel('Error', color=color, fontweight='bold')
ax1.plot(names, mse_scores, marker='o', color=color, label='MSE')
ax1.plot(names, mae_scores, marker='o', linestyle='--', color='purple', label='MAE')  # Changed color to purple
ax1.plot(names, rmse_scores, marker='o', linestyle='--', color='orange', label='RMSE')
ax1.tick_params(axis='y', labelcolor=color)

# Create a secondary y-axis for R2 Score
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('R2 Score', color=color, fontweight='bold')
ax2.plot(names, r2_scores, marker='s', color=color, label='R2 Score')
ax2.tick_params(axis='y', labelcolor=color)

# Title
plt.title('Performance Comparison of Regression Models')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show legend
fig.legend(loc='upper right')

# Show plot
plt.tight_layout()
plt.show()


# # Model Selection

# In[66]:


#select model

dtr=DecisionTreeRegressor()
dtr.fit(x_train_dummy,y_train)
dtr.predict(x_test_dummy)


# # prediction system

# In[71]:




def prediction(Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item):
    features = np.array([[Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item]])                                                                           
        
    transformed_features = preprocesser.transform(features)
    predicted_value=dtr.predict(transformed_features).reshape(1,-1)
    return predicted_value[0]  


# In[72]:


df.head(2)


# In[75]:


Year=1990
average_rain_fall_mm_per_year=1485.0
pesticides_tonnes=121.0
avg_temp=16.37
Area='Albania'
Item="Maize"
prediction(Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item)


# In[77]:


Year=1990
average_rain_fall_mm_per_year=1485.0
pesticides_tonnes=121.0
avg_temp=16.37
Area='Albania'
Item="Potatoes"

prediction(Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item)


# In[ ]:





# In[ ]:




