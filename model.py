import csv

import tensorflow as tf
from pathlib import Path
from keras import layers, ops, optimizers, metrics, Model, applications, callbacks, utils
import keras
import sys
# Wandb imports tensorflow.keras, so we need to replace it with keras in cases where tensorflow.keres doesnt exist
sys.modules["tensorflow.keras"] = keras
import wandb
from wandb.integration.keras import WandbMetricsLogger
from types import SimpleNamespace
from datetime import datetime
from show_samples import predict_dataset

import data

class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = ops.sum(tf.square(anchor - positive), -1)
        an_distance = ops.sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]

if __name__ == "__main__":
    # Convert to SimpleNamespace if needed
    hyperparameters = SimpleNamespace(
        epochs=50,
        batch_size=16,
        image_dim=(224, 224),
        learning_rate=0.0001,
        limit_images=15,
        num_train_classes=-1,
        num_test_classes=-1,
    )

    # Save information
    model_save_path = Path("saved_models")
    model_save_path.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset = data.get_vggface2_data(hyperparameters = hyperparameters)

    base_cnn = applications.resnet.ResNet50(
        weights="imagenet", input_shape=hyperparameters.image_dim + (3,), include_top=False
    )

    flatten = layers.Flatten()(base_cnn.output)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(256, activation="relu")(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(256)(dense2)

    embedding = Model(base_cnn.input, output, name="Embedding")

    utils.plot_model(embedding, to_file="plots/embedding.png", show_shapes=True)

    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

    anchor_input = layers.Input(name="anchor", shape=hyperparameters.image_dim + (3,))
    positive_input = layers.Input(name="positive", shape=hyperparameters.image_dim + (3,))
    negative_input = layers.Input(name="negative", shape=hyperparameters.image_dim + (3,))

    distances = DistanceLayer()(
        embedding(applications.resnet.preprocess_input(anchor_input)),
        embedding(applications.resnet.preprocess_input(positive_input)),
        embedding(applications.resnet.preprocess_input(negative_input)),
    )

    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

    utils.plot_model(siamese_network, to_file="plots/siamese_network.png", show_shapes=False)

    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=optimizers.Adam(hyperparameters.learning_rate))
    siamese_model.summary(expand_nested=True, show_trainable=True)

    model_callbacks = [callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,       # Increased patience to allow more epochs before stopping
        restore_best_weights=True,
    )]

    model_name = f"DOPPEL_Triplet_Embedding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
    try:
        wandb.init(
            project="DOPPEL",
            name=model_name,
            config={
                "type": "tiplet",
                "epochs": hyperparameters.epochs,
                "batch_size": hyperparameters.batch_size,
                "image_dim": hyperparameters.image_dim,
                "learning_rate": hyperparameters.learning_rate,
                "limit_images": hyperparameters.limit_images,
                "num_train_classes": hyperparameters.num_train_classes,
                "num_test_classes": hyperparameters.num_test_classes,
            })
        model_callbacks.append(WandbMetricsLogger())
    except Exception as e:
        print(f"No wandb callback added. {e}")

    history = siamese_model.fit(train_dataset,
                      epochs=hyperparameters.epochs,
                      validation_data=test_dataset,
                      callbacks=model_callbacks)

    # Save model and training history
    embedding.save(model_save_path / model_name)
    with open(model_save_path / f'training_history_{model_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(history.history.keys())
        # Write data
        writer.writerows(zip(*history.history.values()))

    train_dataset : tf.data.Dataset = train_dataset

    similarity_threshold = predict_dataset(embedding, train_dataset)

    sample = next(iter(train_dataset))
    data.visualize(*sample)


    anchor, positive, negative = sample
    anchor_embedding, positive_embedding, negative_embedding = (
        embedding(applications.resnet.preprocess_input(anchor)),
        embedding(applications.resnet.preprocess_input(positive)),
        embedding(applications.resnet.preprocess_input(negative)),
    )

    cosine_similarity = metrics.CosineSimilarity()

    positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
    print("Positive similarity:", positive_similarity.numpy())

    negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
    print("Negative similarity", negative_similarity.numpy())

