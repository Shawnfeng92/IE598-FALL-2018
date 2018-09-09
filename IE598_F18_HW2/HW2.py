from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data[:, 2:]  # we only take the first two features.
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, clf=tree)
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()


from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import metrics

k_range = range(1,26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='minkowski')
    X_train_std = preprocessing.scale(X_train)
    X_test_std = preprocessing.scale(X_test)
    X_combined_std = preprocessing.scale(X_combined)
    knn.fit(X_train_std, y_train)
#    plot_decision_regions(X_combined_std, y_combined, clf=knn)
#    plt.xlabel('petal length [standardized]')
#    plt.ylabel('petal width [standardized]')
#    plt.legend(loc='upper left')
#    plt.show()
    
    y_pred = knn.predict(X_test_std)
    scores.append(metrics.accuracy_score(y_test, y_pred))

print(scores)

knn = KNeighborsClassifier(n_neighbors=6, p=2, metric='minkowski')
X_train_std = preprocessing.scale(X_train)
X_test_std = preprocessing.scale(X_test)
X_combined_std = preprocessing.scale(X_combined)
knn.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, clf=knn)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()


print("My name is {Xiaokang Feng}")
print("My NetID is: {668129807}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
