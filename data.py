import os
import random
from tqdm import tqdm
import tarfile
from operator import contains

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
def generate_triplet(class_to_images):
    print("Generating image pairs...")
    anchor = []
    positive = []
    negative = []

    classes = list(class_to_images.keys())

    for class_name, images in tqdm(class_to_images.items(), desc="Processing classes"):
        # Generate positive pairs
        if len(images) > 1:
            base_image = random.choice(images)
            for i in range(1, len(images)):
                anchor.append(base_image)
                positive.append(images[i])

        # Generate negative pairs
        other_classes = [cls for cls in classes if cls != class_name]

        for i in range(len(images)):
            negative.append(random.choice(class_to_images[random.choice(other_classes)]))

    return anchor, positive, negative

# Load and preprocess a pair of images lazily
def load_image(image_path, image_size):
    image = tf.io.read_file(image_path)
    # Decode image
    image = tf.cond(
        tf.image.is_jpeg(image),
        lambda: tf.image.decode_jpeg(contents=image, channels=3),
        lambda: tf.image.decode_png(contents=image, channels=3))
    image = tf.image.resize(image, image_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def load_triplet(anchor, positive, negative, image_size):
    return load_image(anchor, image_size), load_image(positive, image_size), load_image(negative, image_size)

# Prepare TensorFlow dataset
def create_tf_dataset(anchor, positive, negative, image_size, batch_size):
    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive)
    negative_dataset = tf.data.Dataset.from_tensor_slices(negative)

    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.map(
        lambda anchor, positive, negative : load_triplet(anchor, positive, negative, image_size),
        num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def get_dataset_from_slices(anchor, positive, negative, image_size, batch_size):
    return create_tf_dataset(anchor, positive, negative, image_size, batch_size)

# Load data
def load_data(data_dir, image_size, batch_size):
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

    anchor, positive, negative = generate_triplet(class_to_images)
    return get_dataset_from_slices(anchor, positive, negative, image_size, batch_size)


def get_vggface2_data(data_dir="data/VGG-Face2/data",
                      image_size=(224, 224),
                      random_seed=42,
                      batch_size=32):
    # Main script
    train_dataset = load_data(
        os.path.join(
            data_dir,
            [x for x in os.listdir(data_dir) if contains(x, "train.tar")][0]),
        image_size,
        batch_size
    )
    test_dataset = load_data(
        os.path.join(
            data_dir,
            [x for x in os.listdir(data_dir) if contains(x, "test.tar")][0]),
        image_size,
        batch_size
    )

    print("Training and testing datasets are ready for Siamese network training.")

    return train_dataset, test_dataset

if __name__ == "__main__":
    get_vggface2_data()