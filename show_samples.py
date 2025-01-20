import os
from operator import contains
from types import SimpleNamespace

import click
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models
from model import euclidean_distance, get_contrastive_loss
from datetime import datetime
from data import load_vggface2_folder

def load_latest_model(model_dir) -> models.Model:
    """
    Load the latest model from the specified directory based on the naming convention.

    Parameters:
        model_dir (str): Directory containing the saved models or the specific model to load.

    Returns:
        model: The latest loaded model.
    """
    if os.path.isdir(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.startswith("DOPPEL_Contrastive_Embedding_") and f.endswith(".keras")]
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}. Ensure the models are saved with the correct naming convention.")

        latest_model_file = max(model_files, key=lambda f: datetime.strptime(f.split('_')[3] + '_' + f.split('_')[4].split('.')[0], '%Y%m%d_%H%M%S'))
        model_dir = os.path.join(model_dir, latest_model_file)

    print(f"Loading model from {model_dir}...")
    return models.load_model(model_dir, custom_objects={"contrastive_loss": get_contrastive_loss(1.0)})

def plot_samples_with_predictions(model, dataset, num_samples=5):
    """
    Plot pairs of images with predictions and ground truth labels.

    Parameters:
        model: Trained model to make predictions.
        dataset: Dataset containing the images and labels.
        num_samples: Number of samples to visualize.
    """

    # Get predictions
    sample_images, sample_labels = next(iter(dataset))
    predictions = model.predict(sample_images)

    # Plot the samples
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
    for i, (img1, img2, true_label, pred) in enumerate(zip(*sample_images, sample_labels, predictions)):
        # Show first image
        axes[i, 0].imshow(img1.numpy())
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"Image 1 (Pair {i+1})")

        # Show second image
        axes[i, 1].imshow(img2.numpy())
        axes[i, 1].axis("off")
        axes[i, 1].set_title(f"Image 2 (Pair {i+1})\nPred: {pred[0]:.2f}, True: {true_label}")

    plt.tight_layout()
    plt.savefig('plots/predictions.png')
    plt.show()


@click.command()
@click.option('--data_dir', default="data/VGG-Face2/data", help='Directory containing the dataset.', type=click.Path())
@click.option('--image_size', default=(224, 224), help='Image dimensions.', type=(int, int))
@click.option('--num_pairs', default=4, help='Number of pairs to generate and plot.')
@click.option('--model_path', default="saved_models/DOPPEL_Contrastive_Embedding_20250119_211235.keras", help='Path to the saved models.', type=click.Path())
def main(data_dir, image_size, num_pairs, model_path):
    # Load data
    print("Loading data...")
    hyperparameters = SimpleNamespace(
        image_dim=image_size,
        batch_size=num_pairs,
        num_test_classes=num_pairs,
        limit_images=1,
    )
    dataset = load_vggface2_folder(
        os.path.join(
            data_dir,
            [x for x in os.listdir(data_dir) if contains(x, "test.tar")][0]),
        hyperparameters.image_dim,
        hyperparameters.batch_size,
        hyperparameters.num_test_classes,
        hyperparameters.limit_images
    )

    # Load the Siamese model
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        siamese_model = load_latest_model(model_path)
    else:
        raise FileNotFoundError(f"Model not found at {model_path}. Ensure the model is trained and saved.")

    # Plot samples with predictions
    print("Plotting samples...")
    plot_samples_with_predictions(siamese_model, dataset, num_samples=num_pairs)
    print("Visualization complete.")

if __name__ == "__main__":
    main()