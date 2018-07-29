import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('mnist_test.csv')

label = df['label']
data = df.drop('label', axis = 1)

# plt.figure(figsize=(7,7))
# idx = 500

# grid_data = data.iloc[idx].as_matrix().reshape(28,28)
# plt.imshow(grid_data, interpolation = 'none', cmap = 'gray')
# plt.show()

# print(label[idx])

labels = label.head(10000)
data = data.head(10000)

from sklearn.preprocessing import StandardScaler

standardized_data = StandardScaler().fit_transform(data)

data_5000 = standardized_data[0:10000,:]
label_5000 = labels[0:10000]

model = TSNE(n_components = 2 , random_state = 0)

tsne_data = model.fit_transform(data_5000)

tsne_data = np.vstack((tsne_data.T, label_5000)).T
tsne_df = pd.DataFrame(tsne_data, columns = ['Dimension 1', 'Dimension 2', 'Label'])


sns.FacetGrid(tsne_df, hue = "Label", size = 6).map(plt.scatter,'Dimension 1','Dimension 2', 'Label')
plt.show()


