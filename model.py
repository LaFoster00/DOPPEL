import csv

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from keras import applications
from keras import layers
from keras import losses
from keras import ops
from keras import optimizers
from keras import metrics
from keras import Model
from keras import applications
import keras
import sys
sys.modules["tensorflow.keras"] = keras
import wandb
from wandb.integration.keras import WandbMetricsLogger
from types import SimpleNamespace

import data

# Convert to SimpleNamespace if needed
hyperparameters = SimpleNamespace(
    epochs=10,
    batch_size=32,
    image_dim=(224, 224),
    learning_rate=0.0001,
    num_train_classes=100,
    num_test_classes=10
)

# Save information
model_save_path = Path("saved_models")
model_save_path.mkdir(parents=True, exist_ok=True)

train_dataset, test_dataset = data.get_vggface2_data(hyperparameters = hyperparameters,
                                                     num_train_classes=hyperparameters.num_train_classes,
                                                     num_test_classes=hyperparameters.num_test_classes)

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

trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable

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

siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(hyperparameters.learning_rate))
siamese_model.summary(expand_nested=True, show_trainable=True)

model_callbacks = []

try:
    wandb.init(
        project="DOPPEL",
        config={
            "epochs": hyperparameters.epochs,
            "batch_size": hyperparameters.batch_size,
            "learning_rate": hyperparameters.learning_rate,
            "image_dim": hyperparameters.image_dim,
        })
    model_callbacks.append(WandbMetricsLogger())
except Exception as e:
    print(f"No wandb callback added. {e}")

history = siamese_model.fit(train_dataset,
                  epochs=10,
                  validation_data=test_dataset,
                  callbacks=model_callbacks)

embedding.save(model_save_path / "DOPPEL_Embedding.keras")
with open(model_save_path / 'training_history_doppel.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(history.history.keys())
    # Write data
    writer.writerows(zip(*history.history.values()))

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
