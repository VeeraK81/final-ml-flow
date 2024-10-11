import pandas as pd
import numpy as np
import mlflow
import time
#from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# ---- New libraries ----
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf


# Load data
def load_data(url):
    """
    Instead of loading a CCS file to a dataframe, this function returns a TF dataset 
    """

    return eval(url)

# Preprocess data
def preprocess_data(tf_ds, NUM_CLASSES=10):
    """
    Returns train and test datasets from the TF dataset
    """

    # Train and test datasets
    train_dataset, test_dataset = [tf.data.Dataset.from_tensor_slices(tup).map(lambda image, label: 
                                (tf.convert_to_tensor(tf.expand_dims(image, -1)), 
                                int(tf.keras.utils.to_categorical(label, NUM_CLASSES)))
                                                                            ) for tup in tf_ds]

    return train_dataset, test_dataset, NUM_CLASSES

# # Create the pipeline
# def create_pipeline():
#     """
#     Create a machine learning pipeline with StandardScaler and RandomForestRegressor.

#     Returns:
#         Pipeline: A scikit-learn pipeline object.
#     """
#     return Pipeline(steps=[
#         ("standard_scaler", StandardScaler()),
#         ("Random_Forest", RandomForestRegressor())
#     ])

# Train model
def train_model(train_ds, test_ds, NUM_CLASSES, parameters, verbose= 0):
    """
    Train the CNN model 

    Returns: a TF model object
    """

    INPUT_SHAPE = (28,28,1) 

    FILTERS = parameters['filters']
    BATCH_SIZE = parameters['batch_size']
    EPOCHS = parameters['epochs']

    model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='categorical_accuracy', min_delta=0.001, 
                                                            patience=1, restore_best_weights=True, verbose=verbose)

    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=INPUT_SHAPE, dtype=tf.float32, name="Input"),
        tf.keras.layers.Convolution2D(filters=FILTERS, kernel_size=(5, 5), activation='relu', name='Convolution'),
        tf.keras.layers.MaxPool2D(pool_size=3, name='Max_Pooling'),
        tf.keras.layers.Flatten(name='Flattening'),
        tf.keras.layers.Dense(units=256, activation='relu', name='Dense_1'),
        tf.keras.layers.Dropout(rate=0.1, name="Dropout_1"),
        tf.keras.layers.Dense(units=32, activation='relu', name='Dense_2'),
        tf.keras.layers.Dropout(rate=0.1, name="Dropout_2"),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(train_ds.batch(batch_size=BATCH_SIZE), 
            epochs = EPOCHS,
            validation_data = test_ds.batch(BATCH_SIZE), 
            callbacks = [model_early_stopping],
            verbose = verbose)

    return model

# Log metrics and model to MLflow
def log_metrics_and_model(model, test_ds, artifact_path, registered_model_name, verbose=0):
    """
    Log training and test metrics, and the model to MLflow.

    Args:
        model (CNN w/ TF): The trained model.
        test_ds (pd.DataFrame): Test dataset.
        artifact_path (str): Path to store the model artifact.
        registered_model_name (str): Name to register the model under in MLflow.
    """

    final_val_loss, final_val_categorical_accuracy = model.evaluate(test_ds.batch(1), verbose=verbose)

    mlflow.log_metric("The final out_of_sample_categorical accuracy", round(100 * final_val_categorical_accuracy, 2))

    predictions = model.predict(test_ds.batch(1), verbose=verbose)
    pickle_file = 'confusion_matrix.pkl'
    pd.DataFrame([(label.argmax(), prediction.argmax()) for (image,label), 
                                     prediction in zip(test_ds.as_numpy_iterator(), predictions)], columns=['True', 'Predicted']
                ).groupby(['True', 'Predicted']).size().unstack('Predicted',fill_value=0).to_pickle(pickle_file)
    
    mlflow.log_artifact(pickle_file)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name
    )

# Main function to execute the workflow
def run_experiment(experiment_name, data_url, artifact_path, registered_model_name):
    """
    Run the entire ML experiment pipeline.

    Args:
        experiment_name (str): Name of the MLflow experiment.
        data_url (str): URL to load the TF dataset.
        artifact_path (str): Path to store the model artifact.
        registered_model_name (str): Name to register the model under in MLflow.
    """
    # Start timing
    start_time = time.time()

    # Load and preprocess data
    tf_dataset = load_data(url=data_url)
    train_dataset, test_dataset, NUM_CLASSES = preprocess_data(tf_ds=tf_dataset)

    # Create pipeline
    #pipe = create_pipeline()

    # Set experiment's info 
    mlflow.set_experiment(experiment_name)

    # Get our experiment info
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Call mlflow autolog
    mlflow.sklearn.autolog()

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # Train model
        train_model(train_dataset, test_dataset, parameters, NUM_CLASSES)

    # Print timing
    print(f"...Training Done! --- Total training time: {time.time() - start_time} seconds")

# Entry point for the script
if __name__ == "__main__":
    # Define experiment parameters
    experiment_name = "final_project"
    data_url = 'tf.keras.datasets.mnist.load_data()'
    parameters = {
        'filters':6,
        'batch_size':32,
        'epochs':4
    }
    artifact_path = "output_files"
    registered_model_name = "MNIST-MODEL"

    # Run the experiment
    run_experiment(experiment_name, data_url, parameters, artifact_path, registered_model_name)
