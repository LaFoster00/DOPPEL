import os
import random
from types import SimpleNamespace

from tqdm import tqdm
import tarfile
from operator import contains
import matplotlib.pyplot as plt

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
def generate_triplet(class_to_images, num_classes=-1, max_images=-1):
    print("Generating image pairs...")
    anchor = []
    positive = []
    negative = []

    classes = list(class_to_images.keys())

    if num_classes == -1:
        num_classes = len(classes)

    finished_classes = 0
    for class_name, images in tqdm(class_to_images.items(), desc="Processing classes"):
        if finished_classes == num_classes:
            break
        finished_classes += 1

        num_positive_pairs = 0
        # Generate positive pairs for multiple anchor and comparison images
        if len(images) > 0:
            # for base in range(min(len(images), len(images) if max_images == -1 else max_images)):
            base_image = random.choice(images)
            # Generate multiple
            for comp in range(1, min(len(images), len(images) if max_images == -1 else max_images)):
                anchor.append(base_image)
                positive.append(images[comp])
                num_positive_pairs += 1

        # Generate negative pairs
        other_classes = [cls for cls in classes if cls != class_name]

        # Produce as many negative pairs as positive pairs
        for i in range(num_positive_pairs):
            negative.append(random.choice(class_to_images[random.choice(other_classes)]))

    return anchor, positive, negative


# Load and preprocess a pair of images lazily
def load_image(image_path, image_size, augment=True):
    image = tf.io.read_file(image_path)
    # Decode image
    image = tf.cond(
        tf.image.is_jpeg(image),
        lambda: tf.image.decode_jpeg(contents=image, channels=3),
        lambda: tf.image.decode_png(contents=image, channels=3))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size)

    if augment:
        image = tf.image.random_flip_left_right(image)  # Random horizontal flip
        image = tf.image.random_brightness(image, max_delta=0.2)  # Random brightness adjustment
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Random contrast adjustment
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)  # Random saturation adjustment

    image = tf.clip_by_value(image, 0, 1)

    return image


def load_triplet(anchor, positive, negative, image_size):
    return load_image(anchor, image_size), load_image(positive, image_size), load_image(negative, image_size)


# Prepare TensorFlow dataset
def create_tf_dataset(anchor, positive, negative, image_size, batch_size):
    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive)
    negative_dataset = tf.data.Dataset.from_tensor_slices(negative)

    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = dataset.shuffle(buffer_size=1024*8)
    dataset = dataset.map(
        lambda anchor, positive, negative: load_triplet(anchor, positive, negative, image_size),
        num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# Load data
def load_vggface2_folder(data_dir, image_size, batch_size, num_classes, max_images):
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

    anchor, positive, negative = generate_triplet(class_to_images, num_classes, max_images)
    return create_tf_dataset(anchor, positive, negative, image_size, batch_size)


def get_vggface2_data(hyperparameters,
                      data_dir="data/VGG-Face2/data"):
    # Main script
    train_dataset = load_vggface2_folder(
        os.path.join(
            data_dir,
            [x for x in os.listdir(data_dir) if contains(x, "train.tar")][0]),
        hyperparameters.image_dim,
        hyperparameters.batch_size,
        hyperparameters.num_train_classes,
        hyperparameters.limit_images
    )
    test_dataset = load_vggface2_folder(
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


def visualize(anchor, positive, negative):
    """Visualize a few triplets from the supplied batches."""

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative[i])

    plt.show()


if __name__ == "__main__":
    hyperparameters = SimpleNamespace(
        epochs=50,
        batch_size=16,
        image_dim=(224, 224),
        learning_rate=0.0001,
        limit_images=5,
        num_train_classes=1000,
        num_test_classes=200,
    )
    train_dataset, test_dataset = get_vggface2_data(hyperparameters)
    visualize(*list(train_dataset.take(1).as_numpy_iterator())[0])
    print("Done")
