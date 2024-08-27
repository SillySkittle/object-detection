import os
import cv2
import argparse
from collections import defaultdict
import shutil

# Step 1: Count the number of instances per class in each label file
def count_classes_in_file(file_path):
    class_counts = defaultdict(int)
    with open(file_path, 'r') as file:
        for line in file:
            class_id = int(line.split()[0])
            class_counts[class_id] += 1
    return class_counts


# Step 2: Copy files to output directories
def copy_files(labels_dir, images_dir, output_labels_dir, output_images_dir):
    os.makedirs(output_labels_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)

    for file_name in os.listdir(labels_dir):
        if file_name.endswith('.txt'):
            shutil.copy(os.path.join(labels_dir, file_name), os.path.join(output_labels_dir, file_name))

    for file_name in os.listdir(images_dir):
        if file_name.endswith(('.jpg', '.png')):  # Adjust extensions as needed
            shutil.copy(os.path.join(images_dir, file_name), os.path.join(output_images_dir, file_name))

# Step 3: Perform augmentation if conditions are met
def augment_data(labels_dir, images_dir, classes_file, output_labels_dir, output_images_dir, ratio_threshold):
    # Copy files to output directories
    copy_files(labels_dir, images_dir, output_labels_dir, output_images_dir)


    # Load class names
    with open(classes_file, 'r') as file:
        class_names = [line.strip() for line in file.readlines()]

    # Count total annotations per class across all files
    total_class_counts = defaultdict(int)
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            label_file_path = os.path.join(labels_dir, label_file)
            class_counts = count_classes_in_file(label_file_path)
            for class_id, count in class_counts.items():
                total_class_counts[class_id] += count

    # Determine the class with the highest number of annotations
    max_class_id = max(total_class_counts, key=total_class_counts.get)
    max_class_count = total_class_counts[max_class_id]

    print(f"Class with the most annotations: {class_names[max_class_id]} ({max_class_count} annotations)")

    # Select classes with significantly fewer annotations than the max class
    target_class_ids = {class_id for class_id, count in total_class_counts.items() if
                        count < max_class_count * ratio_threshold}

    # Prompt the user for additional class IDs if desired
    additional_classes = input(
        f"Classes with extremely low data for augmentation: {[class_names[i] for i in target_class_ids]}. Would you like to add any other classes for augmentation? (Enter class names separated by commas or press Enter to skip): ")

    if additional_classes:
        additional_class_ids = {class_names.index(class_name.strip()) for class_name in additional_classes.split(',')}
        target_class_ids.update(additional_class_ids)
    else:
        target_class_ids = target_class_ids

    print(f"Classes selected for augmentation: {[class_names[i] for i in target_class_ids]}")

    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            label_file_path = os.path.join(labels_dir, label_file)
            class_counts = count_classes_in_file(label_file_path)

            # Augment if the file contains any of the target classes
            if any(class_id in target_class_ids for class_id in class_counts):
                # Read the corresponding image
                image_name = label_file.replace('.txt', '.jpg')  # Adjust extension if needed
                image_path = os.path.join(images_dir, image_name)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Image {image_name} not found.")
                    continue

                # Perform horizontal flip on the image
                flipped_image = cv2.flip(image, 1)

                # Save the flipped image
                flipped_image_name = f"f_{image_name}"
                cv2.imwrite(os.path.join(output_images_dir, flipped_image_name), flipped_image)

                # Adjust and save the flipped annotations
                flipped_label_file = os.path.join(output_labels_dir, f"f_{label_file}")
                with open(flipped_label_file, 'w') as out_file:
                    with open(label_file_path, 'r') as in_file:
                        for line in in_file:
                            class_id, x_center, y_center, width, height = map(float, line.split())
                            new_x_center = 1.0 - x_center
                            out_file.write(f"{int(class_id)} {new_x_center} {y_center} {width} {height}\n")

    print("Data augmentation complete. Flipped images and annotations are saved.")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Data augmentation for YOLO datasets.")
    parser.add_argument("labels_dir", type=str, help="Directory containing YOLO label files.")
    parser.add_argument("images_dir", type=str, help="Directory containing image files.")
    parser.add_argument("classes_file", type=str, help="File containing class names.")
    parser.add_argument("output_labels_dir", type=str, help="Directory to save augmented label files.")
    parser.add_argument("output_images_dir", type=str, help="Directory to save augmented image files.")
    parser.add_argument("--ratio_threshold", type=float, default=0.5,
                        help="Ratio threshold to select classes with extremely low data (default: 0.5).")

    args = parser.parse_args()

    # Run the augmentation process
    augment_data(args.labels_dir, args.images_dir, args.classes_file, args.output_labels_dir, args.output_images_dir,
                 args.ratio_threshold)

# python data_augmentation.py yolo_labels images classes.txt augmented_data/all_labels augmented_data/images
# python data_augmentation.py augmented_data/cropped/labels augmented_data/cropped/images classes.txt augmented_data/cropped/aug_crop_data/labels augmented_data/cropped/aug_crop_data/images

