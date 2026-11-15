

import pytest
from model import train_model, predict
from sklearn.datasets import load_iris

# Fixture to create trained model
@pytest.fixture
def model():
    return train_model()


# Test 1: Check model creation
def test_model_creation(model):
    assert model is not None


# Test 2: Check prediction
def test_prediction(model):
    result = predict(model, [[5.1, 3.5, 1.4, 0.2]])
    assert len(result) == 1


# Test 3: Check predicted class
def test_prediction_class(model):
    result = predict(model, [[5.1, 3.5, 1.4, 0.2]])
    assert result[0] in [0, 1, 2]


# Test 4: Check model accuracy
def test_model_accuracy(model):
    iris = load_iris()
    accuracy = model.score(iris.data, iris.target)
    assert accuracy > 0.80


# Test 5: Check multiple predictions
def test_multiple_predictions(model):
    data = [
        [5.1, 3.5, 1.4, 0.2],
        [6.7, 3.0, 5.2, 2.3]
    ]

    result = predict(model, data)
    assert len(result) == 2


# Test 6: Check output data type
def test_output_type(model):
    result = predict(model, [[5.1, 3.5, 1.4, 0.2]])
    assert result.dtype.kind in ['i', 'u']
