import pandas as pd 
from kmodes.kmodes import KModes

train = pd.read_csv('/Users/jessicacichon/Desktop/dsrepo/data-science/customer-segmentation/train_processed.csv')

#Mix of categorical and numerical predictors not appropriate for unsupervised learning algorithms such as KMeans, using KMode instead 
#https://pypi.org/project/kmodes/


km = KModes(n_clusters=4, init='Huang', n_init=5, verbose=1)

clusters = km.fit_predict(train)

# Print the cluster centroids
print(km.cluster_centroids_) 

