import pytest
from unittest import mock
from app.train import load_data, preprocess_data, create_pipeline, train_model

# Test data loading
def test_load_data():
    url = 'tf.keras.datasets.cifar10.load_data()' # A different TF dataset
    assert isinstance(eval(url), tuple), "Dataset is not found"

# Test data preprocessing
def test_preprocess_data():
    url = 'tf.keras.datasets.mnist.load_data()'
    tf_ds = load_data(eval(url))
    train_ds, test_ds, NUM_CLASSES = preprocess_data(tf_ds, NUM_CLASSES=10)
    assert len(train_ds) == 60000, "Training data is of the right size"
    assert len(test_ds) == 10000, "Test data is of the right size"

# Test pipeline creation
# def test_create_pipeline():
#     pipe = create_pipeline()
#     assert "standard_scaler" in pipe.named_steps, "Scaler missing in pipeline"
#     assert "Random_Forest" in pipe.named_steps, "RandomForest missing in pipeline"

# Test model training (mocking GridSearchCV)
# @mock.patch('app.train.GridSearchCV.fit', return_value=None)
def test_train_model(mock_fit):
#    pipe = create_pipeline()
    url = 'tf.keras.datasets.mnist.load_data()'
    train_dataset, test_dataset, NUM_CLASSES = preprocess_data(load_data(url))
    params = {
        'filters':3,
        'batch_size':32,
        'epochs':2
    }
    assert train_model(train_dataset, test_dataset, params, NUM_CLASSES) is not None, "Model training failed"
