import utils
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#Uploading customer training data for EDA
cust_data = pd.read_csv('/Users/jessicacichon/Desktop/dsrepo/data-science/customer-segmentation/train.csv')

print(cust_data.info())
print(cust_data.describe())
print("Total number of records in training data: ", len(cust_data))
 #8069 

#Getting a count of null values 
print((cust_data.isnull().sum()/len(cust_data))*100) #Percentage

#Visualizing missing data 

plt.figure(figsize=(18,8))

colours = ['#34495E', 'seagreen'] 
sns.heatmap(cust_data.isnull(), cmap=sns.color_palette(colours)) 

#Variables with missing values: 
#Ever married: 140 (1.7%), Graduated: 78 (1%), Profession: 124 (1.5%), Work_experience: 829 (10.3%), Family_size: 335 (1.2%), Var_1: 76 (0.9%) 

#Getting numerical columns
cust_num = cust_data.select_dtypes(include='number')
print(cust_num.columns)

#cust_num.hist(bins=30, figsize=(15, 10))

#Getting categorical columns 
cust_cat = cust_data.select_dtypes(include='object')
print(cust_cat.columns)

[print(f'Value counts for column:', cust_cat[col].value_counts()) for col in list(cust_cat.columns)]

#Getting a list of columns with missing values: 
#print([col for col in list(cust_cat.columns) if cust_cat[col].isnull().sum() > 0])

#Imputing categorical variables with the mode value/most frequent for that variable  

cat_impute = cust_cat.columns[cust_cat.isnull().any()].tolist()

for col in cat_impute: 
    cust_data[col].fillna(cust_data[col].mode()[0], inplace=True) #Train data 

#Imputing numerical variables with the mean for that variable  

num_impute = cust_num.columns[cust_num.isnull().any()].tolist() 

for col in num_impute: 
    cust_data[col].fillna(cust_data[col].mean(), inplace=True) #Train data 

print(cust_data.isnull().sum())

print(cust_data.head())

cust_data.to_csv('/Users/jessicacichon/Desktop/dsrepo/data-science/customer-segmentation/train_processed.csv')

#Smarter imputation method, probablistic potentially?? 