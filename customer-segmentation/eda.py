import utils
import pandas as pd 

cust_data = pd.read_csv('/Users/jessicacichon/Desktop/dsrepo/data-science/customer-segmentation/train.csv')
print(cust_data.describe())