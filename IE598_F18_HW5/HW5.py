import numpy as np

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score

iris = datasets.load_iris()
X = iris.data
y = iris.target


dt = DecisionTreeClassifier()

in_sample = []
out_sample = []

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=i+1)
    dt.fit(X_train, y_train)
    in_sample_pred = dt.predict(X_train)
    out_sample_pred = dt.predict(X_test)
    in_sample.append(dt.score(X_train,y_train))
    out_sample.append(dt.score(X_test,y_test))

scores = []
scores.append(in_sample)
scores.append(out_sample)

sd = []
in_sample_sd = np.std(scores[0])
out_sample_sd = np.std(scores[1])
sd.append(in_sample_sd)
sd.append(out_sample_sd)

mean = []
in_sample_mean = np.mean(scores[0])
out_sample_mean = np.mean(scores[1])
mean.append(in_sample_mean)
mean.append(out_sample_mean)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=42)
CV_scores = cross_val_score(dt, X_train, y_train, cv=10, n_jobs=1)
CV_sd = np.std(CV_scores)
CV_mean = np.mean(CV_scores)

CV_acc = dt.score(X_test,y_test)


print("My name is {Xiaokang Feng}")
print("My NetID is: {xf10}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
