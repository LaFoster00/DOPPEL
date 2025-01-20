import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models
from model import euclidean_distance, get_contrastive_loss
from datetime import datetime

def load_latest_model(model_dir):
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

def load_data(data_dir, image_size=(224, 224), num_pairs=5):
    """
    Load images and create equal amounts of matching and non-matching pairs.

    Parameters:
        data_dir (str): Directory containing subfolders for each class.
        image_size (tuple): Target size of the images.
        num_pairs (int): Number of pairs to generate (total matching + non-matching).

    Returns:
        images_1, images_2, labels: Paired images and their labels.
    """
    # Get class directories
    class_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Map each class to its images
    class_to_images = {
        os.path.basename(class_dir): [
            os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith((".jpg", ".jpeg", ".png"))
        ]
        for class_dir in class_dirs
    }

    # Initialize lists for matching and non-matching pairs
    matching_pairs, non_matching_pairs = [], []
    classes = list(class_to_images.keys())

    # Generate equal number of matching pairs (label 1)
    for class_name, images in class_to_images.items():
        if len(images) > 1:
            for _ in range(num_pairs // 2):
                img1, img2 = random.sample(images, 2)
                matching_pairs.append((img1, img2, 1))  # Label 1 for matching pairs

    # Generate equal number of non-matching pairs (label 0)
    for _ in range(num_pairs // 2):
        class1, class2 = random.sample(classes, 2)
        if class_to_images[class1] and class_to_images[class2]:
            img1 = random.choice(class_to_images[class1])
            img2 = random.choice(class_to_images[class2])
            non_matching_pairs.append((img1, img2, 0))  # Label 0 for non-matching pairs

    # Combine and shuffle all pairs
    all_pairs = matching_pairs + non_matching_pairs
    random.shuffle(all_pairs)

    # Load and preprocess images
    images_1 = [tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(p1)), image_size) / 255.0 for p1, _, _ in all_pairs]
    images_2 = [tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(p2)), image_size) / 255.0 for _, p2, _ in all_pairs]
    labels = [label for _, _, label in all_pairs]

    return images_1, images_2, labels


def plot_samples_with_predictions(model, images_1, images_2, labels, num_samples=5):
    """
    Plot pairs of images with predictions and ground truth labels.

    Parameters:
        model: Trained model to make predictions.
        images_1, images_2: Lists of paired images.
        labels: Ground truth labels for the pairs.
        num_samples: Number of samples to visualize.
    """
    # Select equal numbers of matching (label 1) and non-matching (label 0) pairs
    matching_indices = [i for i, label in enumerate(labels) if label == 1]
    non_matching_indices = [i for i, label in enumerate(labels) if label == 0]

    # Randomly sample half from matching and half from non-matching
    selected_matching_indices = random.sample(matching_indices, num_samples // 2)
    selected_non_matching_indices = random.sample(non_matching_indices, num_samples // 2)

    selected_indices = selected_matching_indices + selected_non_matching_indices
    random.shuffle(selected_indices)

    selected_images_1 = [images_1[i] for i in selected_indices]
    selected_images_2 = [images_2[i] for i in selected_indices]
    selected_labels = [labels[i] for i in selected_indices]

    # Get predictions
    predictions = model.predict([tf.stack(selected_images_1), tf.stack(selected_images_2)])

    # Plot the samples
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
    for i, (img1, img2, pred, true_label) in enumerate(zip(selected_images_1, selected_images_2, predictions, selected_labels)):
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


if __name__ == "__main__":
    # Parameters
    DATA_DIR = "data/tmp/VGG-Face2/test"
    IMAGE_SIZE = (224, 224)
    NUM_PAIRS = 4  # This now determines the total number of pairs to generate and plot
    MODEL_PATH = "saved_models"

    # Load data
    print("Loading data...")
    images_1, images_2, labels = load_data(DATA_DIR, image_size=IMAGE_SIZE, num_pairs=NUM_PAIRS)
    print(f"Loaded {len(images_1)} pairs of images.")

    # Load the Siamese model
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        siamese_model = load_latest_model(MODEL_PATH)
    else:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Ensure the model is trained and saved.")

    # Plot samples with predictions
    print("Plotting samples...")
    plot_samples_with_predictions(siamese_model, images_1, images_2, labels, num_samples=NUM_PAIRS)
    print("Visualization complete.")
