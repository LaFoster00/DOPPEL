from operator import contains
from types import SimpleNamespace

import keras
import tqdm
from keras import applications, ops, metrics, models
import tensorflow as tf

from matplotlib import pyplot as plt
from tensorflow import data
import os
from datetime import datetime
from data import load_data, visualize

def load_latest_model(model_dir):
    """
    Load the latest model from the specified directory based on the naming convention.

    Parameters:
        model_dir (str): Directory containing the saved models or the specific model to load

    Returns:
        model: The latest loaded model.
    """
    if os.path.isdir(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.startswith("DOPPEL_Triplet_Embedding_") and f.endswith(".keras")]
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}. Ensure the models are saved with the correct naming convention.")

        latest_model_file = max(model_files, key=lambda f: datetime.strptime(f.split('_')[3] + '_' + f.split('_')[4].split('.')[0], '%Y%m%d_%H%M%S'))
        model_dir = os.path.join(model_dir, latest_model_file)

    print(f"Loading model from {model_dir}...")
    return models.load_model(model_dir)

# Returns the median of the similarity scores of the positive and negative pairs so that we can decide on a threshold
def predict_dataset(embedding : keras.Model, dataset : tf.data.Dataset, similarity_metric=metrics.CosineSimilarity(), batch_size=16, max_iterations=100):
    # Efficient batch processing
    def process_batch(batch):
        anchor, positive, negative = batch

        # Preprocess inputs in batches
        anchor_preprocessed = applications.resnet.preprocess_input(anchor)
        positive_preprocessed = applications.resnet.preprocess_input(positive)
        negative_preprocessed = applications.resnet.preprocess_input(negative)

        # Get embeddings for the batch
        anchor_embedding = embedding(anchor_preprocessed)
        positive_embedding = embedding(positive_preprocessed)
        negative_embedding = embedding(negative_preprocessed)

        # Compute similarities
        positive_similarity = similarity_metric(anchor_embedding, positive_embedding)
        negative_similarity = similarity_metric(anchor_embedding, negative_embedding)

        return ops.mean(positive_similarity), ops.mean(negative_similarity)

    # Batch the dataset
    batch_size = 16
    batched_dataset = dataset.unbatch().batch(batch_size)

    # Iterate through the dataset and process each batch
    total_batches = 0
    positive_similarity, negative_similarity = 0, 0
    prediction_progress = tqdm.tqdm(batched_dataset, desc="Processing batches...")
    for batch in prediction_progress:
        total_batches += 1
        if (total_batches > max_iterations):
            break
        p, n = process_batch(batch)
        positive_similarity += p.numpy()
        negative_similarity += n.numpy()
        prediction_progress.set_description(f"Processing batches... Positive similarity: {positive_similarity / total_batches}, Negative similarity: {negative_similarity / total_batches}, Threshold: {(positive_similarity + negative_similarity) / (2 * total_batches)}")

    return (positive_similarity + negative_similarity) / (2 * total_batches)

def plot_samples_with_predictions(model, dataset, num_samples, similarity_threshold):
    """
    Plot triplets of images with predictions and ground truth labels.

    Parameters:
        model: Trained model to make predictions.
        images_1, images_2: Lists of paired images.
        labels: Ground truth labels for the pairs.
        num_samples: Number of samples to visualize.
    """

    anchor_images = []
    positive_images = []
    p_pred = []
    negative_images = []
    n_pred = []

    anchors, positives, negatives = next(iter(dataset))

    anchor_embeddings, positive_embeddings, negative_embeddings = (
        model(applications.resnet.preprocess_input(anchors)),
        model(applications.resnet.preprocess_input(positives)),
        model(applications.resnet.preprocess_input(negatives)),
    )

    cosine_similarity = metrics.CosineSimilarity()

    for i in range(num_samples):
        anchor, positive, negative = anchors[i], positives[i], negatives[i]
        anchor_images.append(anchor)
        positive_images.append(positive)
        negative_images.append(negative)

        anchor_embedding = anchor_embeddings[i]
        positive_embedding = positive_embeddings[i]
        negative_embedding = negative_embeddings[i]

        positive_similarity = cosine_similarity(anchor_embedding, positive_embedding).numpy()
        p_pred.append(positive_similarity)
        negative_similarity = cosine_similarity(anchor_embedding, negative_embedding).numpy()
        n_pred.append(negative_similarity)

    def show(ax, image, name, pred):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"{name} ({'Similar' if pred > similarity_threshold else 'Different'})")#, {pred:.4f})")

    fig = plt.figure(figsize=(3*2, num_samples*2))

    axs = fig.subplots(num_samples, 3)
    for i in range(num_samples):
        show(axs[i, 0], anchor_images[i], "Anchor", True)
        show(axs[i, 1], positive_images[i], "Positive", p_pred[i])
        show(axs[i, 2], negative_images[i], "Negative", n_pred[i])

    plt.show()

if __name__ == "__main__":
    # Parameters
    DATA_DIR = "data/VGG-Face2/data"
    IMAGE_SIZE = (224, 224)
    NUM_PAIRS = 4  # This now determines the total number of pairs to generate and plot
    MODEL_PATH = "saved_models/"

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

    dataset = test_dataset = load_data(
        os.path.join(
            DATA_DIR,
            [x for x in os.listdir(DATA_DIR) if contains(x, "test.tar")][0]),
        hyperparameters.image_dim,
        hyperparameters.batch_size,
        hyperparameters.num_test_classes,
        hyperparameters.limit_images
    )

    # Load the model
    model = load_latest_model(MODEL_PATH)

    similarity_threshold = predict_dataset(model, dataset, max_iterations=10)
    dataset = dataset.unbatch().batch(NUM_PAIRS)
    plot_samples_with_predictions(model, dataset, NUM_PAIRS, similarity_threshold)