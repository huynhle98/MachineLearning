import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
#n_neighbors = 15


iris = datasets.load_iris()
df = pd.read_csv("adaboost.csv")
# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = df.values[:, :2]
y = df.Decision



positives = df[df['Decision'] >= 0]
negatives = df[df['Decision'] < 0]
"""
plt.scatter(positives['x1'], positives['x2'], marker='+', s=100 * abs(positives['Decision']), c='blue')
plt.scatter(negatives['x1'], negatives['x2'], marker='_', s=100 * abs(negatives['Decision']), c='red')

plt.show()"""

## Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test
h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=400,
                         learning_rate=0.25)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot

Z = Z.reshape(xx.shape)

plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('x1')
plt.ylabel('x2')
plt.title("Adaboost with (n_estimators = %i, alpha = '%.3f')"
          % (abc.n_estimators, abc.learning_rate))
plt.show()