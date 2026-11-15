
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Function to train the model
def train_model():

    # Load Iris dataset
    iris = load_iris()

    # Separate input and output
    X = iris.data
    y = iris.target

    # Create machine learning pipeline
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=200)
    )

    # Train the model
    model.fit(X, y)

    return model


# Function to predict output
def predict(model, data):

    result = model.predict(data)

    return result
