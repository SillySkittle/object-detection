import os
from PIL import Image
import shutil


def check_images_and_labels(image_dir, label_dir, corrupt_image_dir, corrupt_label_dir):
    corrupt_images = []
    corrupt_labels = []

    # Create directories for corrupt files if they don't exist
    os.makedirs(corrupt_image_dir, exist_ok=True)
    os.makedirs(corrupt_label_dir, exist_ok=True)

    # Check images for corruption
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)

        try:
            img = Image.open(img_path)
            img.verify()  # Verify if the image is valid
        except (IOError, SyntaxError) as e:
            corrupt_images.append(img_file)
            shutil.copy(img_path, os.path.join(corrupt_image_dir, img_file))  # Copy corrupt image
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(corrupt_label_dir, label_file))  # Copy corresponding label

    # Check labels for corruption and out-of-bounds coordinates
    for lbl_file in os.listdir(label_dir):
        lbl_path = os.path.join(label_dir, lbl_file)
        img_file = os.path.splitext(lbl_file)[0] + '.jpg'  # Assuming the image extension is .jpg
        img_path = os.path.join(image_dir, img_file)

        try:
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    values = line.strip().split()
                    if len(values) != 5:
                        raise ValueError("Incorrect label format")

                    # Check if the coordinates are normalized and within bounds [0, 1]
                    _, x_center, y_center, width, height = map(float, values)
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                        raise ValueError("Coordinates out of bounds")

        except (IOError, ValueError) as e:
            corrupt_labels.append(lbl_file)
            shutil.copy(lbl_path, os.path.join(corrupt_label_dir, lbl_file))  # Copy corrupt label
            if os.path.exists(img_path):
                shutil.copy(img_path, os.path.join(corrupt_image_dir, img_file))  # Copy corresponding image

    return corrupt_images, corrupt_labels


image_dir = 'datasets/augmented_data/cropped/images'
label_dir = 'datasets/augmented_data/cropped/labels'  # Change to your label directory
corrupt_image_dir = 'datasets/augmented_data/cropped/corrupt_images'  # Change to your corrupt images directory
corrupt_label_dir = 'datasets/augmented_data/cropped/corrupt_labels'  # Change to your corrupt labels directory

corrupt_images, corrupt_labels = check_images_and_labels(image_dir, label_dir, corrupt_image_dir, corrupt_label_dir)

print(f"Corrupt images: {corrupt_images}")
print(f"Corrupt labels: {corrupt_labels}")
