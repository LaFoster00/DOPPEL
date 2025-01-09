import os
import random
from types import SimpleNamespace

from tqdm import tqdm
import tarfile
import matplotlib.pyplot as plt
import tensorflow as tf

def extract_tar(tar_path, extract_dir):
    if "train.tar" in tar_path:
        data_type = "train"
    else:
        data_type = "test"

    if not os.path.exists(os.path.join(extract_dir, data_type)):
        print(f"Extracting {data_type} data...")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)

# Lazy image pair generator for contrastive loss
def pair_generator(class_to_images, image_size):
    """
    Generator for creating strictly balanced similar and dissimilar pairs.
    """
    for class_name, images in class_to_images.items():
        if len(images) > 1:
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    # Generate a similar pair (label=0)
                    yield load_image(images[i], image_size), load_image(images[j], image_size), 0
                    
                    # Generate a corresponding dissimilar pair (label=1)
                    random_class = random.choice([cls for cls in class_to_images if cls != class_name])
                    random_image = random.choice(class_to_images[random_class])
                    yield load_image(images[i], image_size), load_image(random_image, image_size), 1

def load_image(image_path, image_size):
    """Load and preprocess an image."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def create_tf_pair_dataset(generator_fn, image_size, batch_size):
    """Create TensorFlow dataset for contrastive loss using a generator."""
    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_signature=(
            tf.TensorSpec(shape=(image_size[0], image_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(image_size[0], image_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    dataset = dataset.shuffle(buffer_size=1024)  # Adjust for memory constraints
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

import os

def load_data_for_contrastive_loss(data_dir="data/VGG-Face2/data", 
                                   hyperparameters=None, 
                                   limit_images=None, 
                                   num_train_classes=None, 
                                   num_test_classes=None):
    """
    Load the data for contrastive loss efficiently.
    
    Parameters:
        data_dir (str): Directory containing the data.
        hyperparameters: Hyperparameters for the data loading process.
        limit_images (int, optional): Maximum number of images to load per class.
        num_train_classes (int, optional): Number of training classes.
        num_test_classes (int, optional): Number of test classes.
        
    Returns:
        train_dataset: TensorFlow dataset for training.
        test_dataset: TensorFlow dataset for testing.
        num_train_classes: Number of training classes.
        num_test_classes: Number of test classes.
    """
    if hyperparameters is None:
        raise ValueError("Hyperparameters must be provided.")

    image_size = getattr(hyperparameters, "image_dim", (224, 224))
    batch_size = getattr(hyperparameters, "batch_size", 32)

    print(f"Loading data from {data_dir} for contrastive loss...")
    extract_tar(data_dir, "data/tmp/VGG-Face2")

    # Define the paths for train and test datasets
    train_data_dir = "data/tmp/VGG-Face2/train"
    test_data_dir = "data/tmp/VGG-Face2/test"

    # Initialize dictionaries to hold class-to-images mappings
    train_class_to_images = {}
    test_class_to_images = {}

    # Collect images per class for training dataset
    all_train_classes = os.listdir(train_data_dir)
    if num_train_classes:
        all_train_classes = all_train_classes[:num_train_classes]  # Limit to num_train_classes
    for class_name in all_train_classes:
        class_path = os.path.join(train_data_dir, class_name)
        if os.path.isdir(class_path):
            images = [
                os.path.join(class_path, img)
                for img in os.listdir(class_path)
                if img.endswith(".jpg")
            ]
            train_class_to_images[class_name] = images[:limit_images] if limit_images else images

    # Collect images per class for testing dataset
    all_test_classes = os.listdir(test_data_dir)
    if num_test_classes:
        all_test_classes = all_test_classes[:num_test_classes]  # Limit to num_test_classes
    for class_name in all_test_classes:
        class_path = os.path.join(test_data_dir, class_name)
        if os.path.isdir(class_path):
            images = [
                os.path.join(class_path, img)
                for img in os.listdir(class_path)
                if img.endswith(".jpg")
            ]
            test_class_to_images[class_name] = images[:limit_images] if limit_images else images

    # Calculate the number of classes
    num_train_classes = len(train_class_to_images)
    num_test_classes = len(test_class_to_images)

    # Create datasets lazily
    train_dataset = create_tf_pair_dataset(
        lambda: pair_generator(train_class_to_images, image_size),
        image_size,
        batch_size,
    )
    test_dataset = create_tf_pair_dataset(
        lambda: pair_generator(test_class_to_images, image_size),
        image_size,
        batch_size,
    )

    return train_dataset, test_dataset


def visualize(dataset):
    """Visualize a few triplets or pairs from the dataset."""
    sample = next(iter(dataset.take(1)))
    image_1, image_2, labels = sample

    fig, axes = plt.subplots(3, 2, figsize=(6, 9))
    for i in range(3):
        axes[i, 0].imshow(image_1[i].numpy())
        axes[i, 1].imshow(image_2[i].numpy())
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')
        axes[i, 0].set_title(f"Label: {labels[i].numpy()}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    hyperparameters = SimpleNamespace(
        epochs=10,
        batch_size=32,
        image_dim=(224, 224),
        learning_rate=0.0003,
    )
    train_dataset, test_dataset = load_data_for_contrastive_loss(hyperparameters=hyperparameters)
    visualize(train_dataset)
    print("Done")
