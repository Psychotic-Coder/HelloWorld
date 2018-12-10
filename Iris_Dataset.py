#Iris Dataset

from sklearn import datasets						#to load iris dataset
from sklearn.model_selection import train_test_split	#to split dataset into train and test sets
from sklearn import metrics							#to calculate accuracy of the model

df = datasets.load_iris()
X = df['data']		#Feature matrix
y = df['target']	#Label vector

Xtrain, Xtest = train_test_split(X, test_size = 0.2, random_state=42)	#20% of data is now test set and rest is train set
ytrain, ytest = train_test_split(y, test_size = 0.2, random_state=42)	#20% of data is now test set and rest is train set

from sklearn.neighbors import KNeighborsClassifier		#KNN model
from sklearn.linear_model import LogisticRegression		#Logistic Regression model

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(Xtrain, ytrain)
pr1 = knn1.predict(X)

knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(Xtrain, ytrain)
pr2 = knn5.predict(X)

clf = LogisticRegression()
clf.fit(Xtrain, ytrain)
pr3 = clf.predict(X)

print('KNN, n=5:', metrics.accuracy_score(y, pr1))
print('KNN, n=1:', metrics.accuracy_score(y, pr2))
print('Logistic Regression:', metrics.accuracy_score(y, pr3))