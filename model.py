import csv
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import keras
from keras import applications, layers, Model, optimizers, metrics, callbacks, regularizers
from types import SimpleNamespace
import data
from math import comb, floor, ceil
import wandb
import sys
from datetime import datetime

sys.modules["tensorflow.keras"] = keras
from wandb.integration.keras import WandbMetricsLogger
from keras import saving

@saving.register_keras_serializable(package='MyPackage')
def euclidean_distance(vectors):
    (a, b) = vectors
    sum_squared = keras.ops.sum(keras.ops.square(a - b), axis=1,
                                       keepdims=True)
    return keras.ops.sqrt(keras.ops.maximum(sum_squared, keras.backend.epsilon()))

def  get_contrastive_loss(margin):
    def contrastive_loss(y_true, y_pred):
        squared_distance = tf.square(y_pred)
        # Compute contrastive loss
        loss = (1 - y_true) * 0.5 * squared_distance + y_true * 0.5 * tf.maximum(0.0, margin - y_pred) ** 2
        return tf.reduce_mean(loss)
    return contrastive_loss

if __name__ == "__main__":
    # Define hyperparameters
    hyperparameters = SimpleNamespace(
        epochs=50,
        batch_size=32,
        image_dim=(224, 224),
        learning_rate=0.0001,
        limit_images=15,
        num_train_classes=-1,
        num_test_classes=-1,
        trainable_layers=20,
        dropout_rate = 0.5,
        margin = 2.0
    )

    # Prepare model save path
    model_save_path = Path("saved_models")
    model_save_path.mkdir(parents=True, exist_ok=True)

    # Load datasets
    #train_dataset, test_dataset = data.load_data_for_contrastive_loss(hyperparameters=hyperparameters, limit_images=hyperparameters.limit_images, num_test_classes=hyperparameters.num_test_classes, num_train_classes=hyperparameters.num_train_classes)
    train_dataset, test_dataset = data.get_vggface2_data(hyperparameters)

    # Create Siamese Network
    image_1_input = layers.Input(name="image_1", shape=hyperparameters.image_dim + (3,))

    image_2_input = layers.Input(name="image_2", shape=hyperparameters.image_dim + (3,))

    # Define base CNN for embeddings
    base_cnn = applications.ResNet50(
        weights="imagenet", input_shape=hyperparameters.image_dim + (3,), include_top=False
    )

    # Make only the last few layers trainable
    trainable_layers = hyperparameters.trainable_layers  # Adjust the number of layers you want to train
    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable


    flatten = layers.GlobalAveragePooling2D()(base_cnn.output)
    dropout1 = layers.Dropout(hyperparameters.dropout_rate)(flatten)
    # Add L2 regularization
    dense1 = layers.Dense(
        1024,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)  # L2 regularization factor = 0.01
    )(dropout1)
    dropout2 = layers.Dropout(hyperparameters.dropout_rate)(dense1)
    output = layers.Dense(
        512,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)  # L2 regularization factor = 0.01
    )(dropout1)
    embedding = Model(base_cnn.input, output, name="Embedding")

    embedding_1 = embedding(image_1_input)
    embedding_2 = embedding(image_2_input)

    distance = layers.Lambda(euclidean_distance, name='dist', output_shape=(1,))([embedding_1, embedding_2])

    output = layers.Dense(1, activation='sigmoid')(distance)

    siamese_model = Model(inputs=[image_1_input, image_2_input], outputs=output, name="SiameseNetwork")

    # Compile and summarize the model
    siamese_model.compile(optimizer=optimizers.Adam(hyperparameters.learning_rate), loss=get_contrastive_loss(hyperparameters.margin), metrics=["accuracy"])
    siamese_model.summary()

    # Define callbacks for training
    model_callbacks = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,       # Increased patience to allow more epochs before stopping
            restore_best_weights=True,
        )
    ]


    model_name = f"DOPPEL_Contrastive_Embedding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
    # Initialize Weights & Biases tracking if available
    try:
        wandb.init(
            project="DOPPEL",
            name=model_name,
            config={
                "epochs": hyperparameters.epochs if hasattr(hyperparameters, 'epochs') else 50,
                "batch_size": hyperparameters.batch_size if hasattr(hyperparameters, 'batch_size') else 32,
                "dropout": hyperparameters.dropout_rate if hasattr(hyperparameters, 'dropout_rate') else 0.5,
                "num_train_classes": hyperparameters.num_train_classes if hasattr(hyperparameters, 'num_train_classes') else 100,
                "num_val_classes": hyperparameters.num_val_classes if hasattr(hyperparameters, 'num_val_classes') else 20,
                "input_image_size": hyperparameters.input_image_size if hasattr(hyperparameters, 'input_image_size') else (224, 224),
                "limit_images": hyperparameters.limit_images,
                "type":"contrastive",
                "trainable_layers": hyperparameters.trainable_layers,
                "dropout_rate": hyperparameters.dropout_rate,
                "margin": hyperparameters.margin
            })
        model_callbacks.append(WandbMetricsLogger())
    except Exception as e:
        print(f"No wandb callback added. Error: {e}")


    # Train the model
    history = siamese_model.fit(
        train_dataset,
        epochs=hyperparameters.epochs,
        validation_data=test_dataset,
        callbacks=model_callbacks
    )

    # Save model and training history
    siamese_model.save(model_save_path / model_name)
    with open(model_save_path / 'training_history_doppel.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(history.history.keys())
        writer.writerows(zip(*history.history.values()))

    # Evaluate and visualize results
    sample = next(iter(train_dataset))
    data.visualize(train_dataset)
    (image_1, image_2), _ = sample
    embedding_1, embedding_2 = (
        embedding(image_1),
        embedding(image_2)
    )
    cosine_similarity = metrics.CosineSimilarity()
    print("Cosine Similarity:", cosine_similarity(embedding_1, embedding_2).numpy())
