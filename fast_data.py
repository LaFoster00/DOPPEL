import os
import random
from types import SimpleNamespace

from tqdm import tqdm
import tarfile
from operator import contains
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf


def extract_tar(tar_path, extract_dir):
    if contains(tar_path, "train.tar"):
        type = "train"
    else:
        type = "test"

    if not os.path.exists(os.path.join(extract_dir, type)):
        print(f"Extracting {type} data...")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)


# Generate image pairs
def generate_pairs(class_to_images, num_classes=-1, max_images=-1):
    print("Generating image pairs...")
    positive_pairs = []
    negative_pairs = []

    classes = list(class_to_images.keys())

    if num_classes == -1:
        num_classes = len(classes)

    finished_classes = 0
    for class_name, images in tqdm(class_to_images.items(), desc="Processing classes"):
        if finished_classes == num_classes:
            break
        finished_classes += 1

        num_positive_pairs = 0
        # Generate positive pairs
        if len(images) > 0:
            base_image = random.choice(images)
            for i in range(1, min(len(images), len(images) if max_images == -1 else max_images)):
                positive_pairs.append((base_image, images[i], 1))
                num_positive_pairs += 1

        # Generate negative pairs
        other_classes = [cls for cls in classes if cls != class_name]

        for i in range(num_positive_pairs):
            base_image = random.choice(images)
            negative_pairs.append((base_image, random.choice(class_to_images[random.choice(other_classes)]), 0))

    return positive_pairs, negative_pairs


# Load and preprocess a pair of images lazily
def load_image(image_path, image_size, augment=True):
    image = tf.io.read_file(image_path)
    # Decode image
    image = tf.cond(
        tf.image.is_jpeg(image),
        lambda: tf.image.decode_jpeg(contents=image, channels=3),
        lambda: tf.image.decode_png(contents=image, channels=3))
    image = tf.image.resize(image, image_size)
    image = image / 255.0  # Normalize to [0, 1]

    if augment:
        image = tf.image.random_flip_left_right(image)  # Random horizontal flip
        image = tf.image.random_brightness(image, max_delta=0.2)  # Random brightness adjustment
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Random contrast adjustment
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)  # Random saturation adjustment

    image = tf.clip_by_value(image, 0, 1)
    #image = image * 255.0
    #image = image - 127.5
    #image = tf.cast(image, tf.uint8)

    return image


def load_pair(base, comp, flag, image_size):
    return load_image(base, image_size), load_image(comp, image_size), flag


# Prepare TensorFlow dataset
def create_tf_dataset(positive_pairs, negative_pairs, image_size, batch_size):
    pairs = positive_pairs + negative_pairs
    image_paths = [(pair[0], pair[1]) for pair in pairs]
    labels = [pair[2] for pair in pairs]

    # Convert to NumPy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)

    # Generate and store shuffling indices
    indices = np.random.permutation(len(image_paths))

    # Apply the same indices to shuffle both arrays
    shuffled_image_paths = image_paths[indices]
    shuffled_labels = labels[indices]

    # Convert back to Python lists if needed
    #image_paths = shuffled_image_paths.tolist()
    #labels = shuffled_labels.tolist()

    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    label_dataset = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    dataset = dataset.shuffle(buffer_size=4096)
    dataset = dataset.map(
        lambda paths, flag: load_pair(paths[0], paths[1], flag, image_size),
        num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# Load data
def load_data(data_dir, image_size, batch_size, num_classes, max_images):
    print(f"Loading data from {data_dir}...")
    extract_tar(data_dir, "data/tmp/VGG-Face2")
    if contains(data_dir, "train.tar"):
        data_dir = "data/tmp/VGG-Face2/train"
    else:
        data_dir = "data/tmp/VGG-Face2/test"
    classes = os.listdir(data_dir)
    class_to_images = {}

    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(".jpg")]
            class_to_images[class_name] = images



    positive_pairs, negative_pairs = generate_pairs(class_to_images, num_classes, max_images)
    return create_tf_dataset(positive_pairs, negative_pairs, image_size, batch_size)


def get_vggface2_data(hyperparameters,
                      data_dir="data/VGG-Face2/data"):
    # Main script
    train_dataset = load_data(
        os.path.join(
            data_dir,
            [x for x in os.listdir(data_dir) if contains(x, "train.tar")][0]),
        hyperparameters.image_dim,
        hyperparameters.batch_size,
        hyperparameters.num_train_classes,
        hyperparameters.limit_images
    )
    test_dataset = load_data(
        os.path.join(
            data_dir,
            [x for x in os.listdir(data_dir) if contains(x, "test.tar")][0]),
        hyperparameters.image_dim,
        hyperparameters.batch_size,
        hyperparameters.num_test_classes,
        hyperparameters.limit_images
    )

    print("Training and testing datasets are ready for Siamese network training.")

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
    os.makedirs("plots", exist_ok=True)
    plt.savefig('plots/sample.png')


if __name__ == "__main__":
    # Convert to SimpleNamespace if needed
    hyperparameters = SimpleNamespace(
        epochs=50,
        batch_size=16,
        image_dim=(224, 224),
        learning_rate=0.0001,
        limit_images=5,
        num_train_classes=1000,
        num_test_classes=200,
        trainable_layers=20,
        dropout_rate=0.5,
        margin=1.0
    )
    train_dataset, test_dataset = get_vggface2_data(hyperparameters)
    visualize(train_dataset)
    print("Done")
