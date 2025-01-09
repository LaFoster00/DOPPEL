import csv
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from keras import applications, layers, Model, optimizers, metrics
from types import SimpleNamespace
import data

# Define hyperparameters
hyperparameters = SimpleNamespace(
    epochs=10,
    batch_size=32,
    image_dim=(224, 224),
    learning_rate=0.0001,
    num_train_classes=-1,
    num_test_classes=-1
)

# Prepare model save path
model_save_path = Path("saved_models")
model_save_path.mkdir(parents=True, exist_ok=True)

# Load datasets
train_dataset, test_dataset = data.load_data_for_contrastive_loss(hyperparameters=hyperparameters)

# Define base CNN for embeddings
base_cnn = applications.ResNet50(
    weights="imagenet", input_shape=hyperparameters.image_dim + (3,), include_top=False
)
flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)
embedding = Model(base_cnn.input, output, name="Embedding")

# Set trainable layers for fine-tuning
trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable

# Custom layer for computing distances
class DistanceLayer(layers.Layer):
    def call(self, embedding_1, embedding_2):
        squared_diff = tf.square(embedding_1 - embedding_2)
        squared_distances = tf.reduce_sum(squared_diff, axis=-1)
        return tf.expand_dims(tf.sqrt(squared_distances), axis=-1)

# Define Siamese Model
class SiameseModel(Model):
    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        image_1, image_2, label = data
        label = tf.cast(label, tf.float32)
        distances = self.siamese_network([image_1, image_2])
        return self._contrastive_loss(distances, label)

    def _contrastive_loss(self, distance, label):
        margin = self.margin
        loss_positive = label * tf.square(distance)
        loss_negative = (1 - label) * tf.square(tf.maximum(0.0, margin - distance))
        return 0.5 * tf.reduce_mean(loss_positive + loss_negative)

    @property
    def metrics(self):
        return [self.loss_tracker]

# Create Siamese Network
image_1_input = layers.Input(name="image_1", shape=hyperparameters.image_dim + (3,))
image_2_input = layers.Input(name="image_2", shape=hyperparameters.image_dim + (3,))
embedding_1 = embedding(applications.resnet.preprocess_input(image_1_input))
embedding_2 = embedding(applications.resnet.preprocess_input(image_2_input))
distances = DistanceLayer()(embedding_1, embedding_2)
siamese_network = Model(inputs=[image_1_input, image_2_input], outputs=distances, name="SiameseNetwork")

# Compile and summarize the model
siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(hyperparameters.learning_rate))
siamese_model.summary()

# Train the model
history = siamese_model.fit(
    train_dataset,
    epochs=hyperparameters.epochs,
    validation_data=test_dataset
)

# Save model and training history
embedding.save(model_save_path / "DOPPEL_Embedding.keras")
with open(model_save_path / 'training_history_doppel.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(history.history.keys())
    writer.writerows(zip(*history.history.values()))

# Evaluate and visualize results
sample = next(iter(train_dataset))
data.visualize(train_dataset)
image_1, image_2, _ = sample
embedding_1, embedding_2 = (
    embedding(applications.resnet.preprocess_input(image_1)),
    embedding(applications.resnet.preprocess_input(image_2))
)
cosine_similarity = metrics.CosineSimilarity()
print("Cosine Similarity:", cosine_similarity(embedding_1, embedding_2).numpy())
