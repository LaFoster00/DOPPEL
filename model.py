import csv
import click
import tensorflow as tf
from pathlib import Path
from keras import layers, ops, optimizers, metrics, Model, applications, callbacks, utils
import keras
import sys
from types import SimpleNamespace
from datetime import datetime
from show_samples import predict_dataset
import data
# Wandb imports tensorflow.keras, so we need to replace it with keras in cases where tensorflow.keres doesnt exist
sys.modules["tensorflow.keras"] = keras
import wandb
from wandb.integration.keras import WandbMetricsLogger

sys.modules["tensorflow.keras"] = keras

class DistanceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = ops.sum(tf.square(anchor - positive), -1)
        an_distance = ops.sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

class SiameseModel(Model):
    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

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
        ap_distance, an_distance = self.siamese_network(data)
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]

def build_embedding_model(image_dim):
    base_cnn = applications.resnet.ResNet50(weights="imagenet", input_shape=image_dim + (3,), include_top=False)
    flatten = layers.Flatten()(base_cnn.output)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(256, activation="relu")(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(256)(dense2)
    embedding = Model(base_cnn.input, output, name="Embedding")
    return embedding

def build_siamese_network(embedding, image_dim):
    anchor_input = layers.Input(name="anchor", shape=image_dim + (3,))
    positive_input = layers.Input(name="positive", shape=image_dim + (3,))
    negative_input = layers.Input(name="negative", shape=image_dim + (3,))
    distances = DistanceLayer()(
        embedding(applications.resnet.preprocess_input(anchor_input)),
        embedding(applications.resnet.preprocess_input(positive_input)),
        embedding(applications.resnet.preprocess_input(negative_input)),
    )
    siamese_network = Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)
    return siamese_network

def set_trainable_layers(base_cnn):
    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

def save_training_history(history, model_save_path, model_name):
    with open(model_save_path / f'training_history_{model_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(history.history.keys())
        writer.writerows(zip(*history.history.values()))

@click.command()
@click.option('--epochs', default=50, help='Number of epochs.')
@click.option('--batch_size', default=16, help='Batch size.')
@click.option('--image_dim', default=(224, 224), help='Image dimensions.', type=(int, int))
@click.option('--learning_rate', default=0.0001, help='Learning rate.')
@click.option('--limit_images', default=15, help='Limit image comparisons per person.')
@click.option('--num_train_classes', default=-1, help='Number of training classes (Persons).')
@click.option('--num_test_classes', default=-1, help='Number of test classes (Persons).')
@click.option('--data_dir', default="data/VGG-Face2/data", help='Path to the VGG-Face2 dataset.', type=click.Path())
def train_model(epochs, batch_size, image_dim, learning_rate, limit_images, num_train_classes, num_test_classes, data_dir):
    hyperparameters = SimpleNamespace(
        epochs=epochs,
        batch_size=batch_size,
        image_dim=image_dim,
        learning_rate=learning_rate,
        limit_images=limit_images,
        num_train_classes=num_train_classes,
        num_test_classes=num_test_classes,
    )

    model_save_path = Path("saved_models")
    model_save_path.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset = data.get_vggface2_data(hyperparameters=hyperparameters, data_dir=data_dir)

    embedding = build_embedding_model(hyperparameters.image_dim)
    set_trainable_layers(embedding)
    siamese_network = build_siamese_network(embedding, hyperparameters.image_dim)

    utils.plot_model(embedding, to_file="plots/embedding.png", show_shapes=True)
    utils.plot_model(siamese_network, to_file="plots/siamese_network.png", show_shapes=False)

    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=optimizers.Adam(hyperparameters.learning_rate))
    siamese_model.summary(expand_nested=True, show_trainable=True)

    model_callbacks = [callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

    model_name = f"DOPPEL_Triplet_Embedding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
    try:
        wandb.init(
            project="DOPPEL",
            name=model_name,
            config={
                "type": "triplet",
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

    history = siamese_model.fit(train_dataset, epochs=hyperparameters.epochs, validation_data=test_dataset, callbacks=model_callbacks)

    embedding.save(model_save_path / model_name)
    save_training_history(history, model_save_path, model_name)

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

if __name__ == "__main__":
    train_model()