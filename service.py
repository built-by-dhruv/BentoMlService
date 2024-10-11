import bentoml
import numpy as np
from bentoml.io import NumpyNdarray

iris_clf_runner = bentoml.sklearn.get("iris_svm_model:latest").to_runner()

svc = bentoml.Service("iris_classifier",runners=[iris_clf_runner])

@svc.api(input = NumpyNdarray(),output =  NumpyNdarray())
def classify(data: np.ndarray):
    # return prediction
    return svc.predict.run(data)