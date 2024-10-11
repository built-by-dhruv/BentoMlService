import bentoml

from sklearn import svm
from sklearn import datasets


# load iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target\


# train a SVM classifier using the iris dataset
clf = svm.SVC(gamma='scale')
clf.fit(X, y)


# Save the model using BentoML
saved_model=bentoml.sklearn.save_model("iris_svm_model", clf)
print(f"{saved_model} saved")
# Model(tag="iris_svm_model:jxemmc4iesw7kxdb")