import csv
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from keras import applications, layers, Model, optimizers, metrics, utils
from types import SimpleNamespace
import data
from math import comb, floor, ceil

# Define hyperparameters
hyperparameters = SimpleNamespace(
    epochs=10,
    batch_size=16,
    image_dim=(224, 224),
    learning_rate=0.0001,
    limit_images=5,
    num_train_classes=800,
    num_test_classes=200
)

num_combinations = comb(hyperparameters.limit_images, 2)

# Calculate steps per epoch
steps_per_epoch = ceil(
    (num_combinations * hyperparameters.num_train_classes * 2)
    / hyperparameters.batch_size
)

validation_steps = ceil(
    (num_combinations * hyperparameters.num_test_classes * 2)
    / hyperparameters.batch_size
)

# Prepare model save path
model_save_path = Path("saved_models")
model_save_path.mkdir(parents=True, exist_ok=True)

# Load datasets
train_dataset, test_dataset = data.load_data_for_contrastive_loss(hyperparameters=hyperparameters, limit_images=hyperparameters.limit_images, num_test_classes=hyperparameters.num_test_classes, num_train_classes=hyperparameters.num_train_classes)

def euclidean_distance(vectors):
    (a, b) = vectors
    sum_squared = tf.keras.backend.sum(tf.keras.backend.square(a - b), axis=1,
                                       keepdims=True)
    return tf.keras.backend.sqrt(tf.keras.backend.maximum(sum_squared, tf.keras.backend.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1.0):

    squared_distance = tf.square(y_pred)

    # Compute contrastive loss
    loss = (1 - y_true) * 0.5 * squared_distance + y_true * 0.5 * tf.maximum(0.0, margin - y_pred) ** 2

    return tf.reduce_mean(loss)

# Create Siamese Network
image_1_input = layers.Input(name="image_1", shape=hyperparameters.image_dim + (3,))
image_2_input = layers.Input(name="image_2", shape=hyperparameters.image_dim + (3,))

# Define base CNN for embeddings
base_cnn = applications.ResNet50(
    weights="imagenet", input_shape=hyperparameters.image_dim + (3,), include_top=False
)

flatten = layers.GlobalAveragePooling2D()(base_cnn.output)
output = layers.Dense(1024, activation='relu')(flatten)
embedding = Model(base_cnn.input, output, name="Embedding")

embedding_1 = embedding(image_1_input)
embedding_2 = embedding(image_2_input)

distance = layers.Lambda(euclidean_distance, name='dist')([embedding_1, embedding_2])

output = layers.Dense(1, activation='sigmoid')(distance)

siamese_model = Model(inputs=[image_1_input, image_2_input], outputs=distance, name="SiameseNetwork")

# Compile and summarize the model
siamese_model.compile(optimizer=optimizers.Adam(hyperparameters.learning_rate), loss=contrastive_loss, metrics=["accuracy"])
siamese_model.summary()

# Train the model
history = siamese_model.fit(
    train_dataset.map(lambda image_1, image_2, label: ((image_1, image_2), label)).repeat(),  # Pack images into tuple for two inputs
    epochs=hyperparameters.epochs,
    validation_data=test_dataset.map(lambda image_1, image_2, label: ((image_1, image_2), label)).repeat(),  # Same for validation
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

# Save model and training history
siamese_model.save(model_save_path / "DOPPEL_Embedding.keras")
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
