import os
import shutil
from sklearn.model_selection import train_test_split
import argparse

def split_dataset(images_dir, annotations_dir, output_dir, test_size=0.2):  # get all image files

    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

    train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=42) # Split into train and test

    os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'labels'), exist_ok=True)

    for file_name in train_files:  # copy train files
        shutil.copy(os.path.join(images_dir, file_name), os.path.join(output_dir, 'train', 'images', file_name))
        annotation_file = os.path.splitext(file_name)[0] + ".txt"
        if os.path.exists(os.path.join(annotations_dir, annotation_file)):
            shutil.copy(os.path.join(annotations_dir, annotation_file), os.path.join(output_dir, 'train', 'labels', annotation_file))

    for file_name in test_files:  # copy test files
        shutil.copy(os.path.join(images_dir, file_name), os.path.join(output_dir, 'test', 'images', file_name))
        annotation_file = os.path.splitext(file_name)[0] + ".txt"
        if os.path.exists(os.path.join(annotations_dir, annotation_file)):
            shutil.copy(os.path.join(annotations_dir, annotation_file), os.path.join(output_dir, 'test', 'labels', annotation_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train and test sets.")
    parser.add_argument("images_dir", help="Directory containing images.")
    parser.add_argument("annotations_dir", help="Directory containing annotations.")
    parser.add_argument("output_dir", help="Directory to save split datasets.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size.")
    args = parser.parse_args()

    split_dataset(args.images_dir, args.annotations_dir, args.output_dir, args.test_size)

# python datasets/split_dataset.py datasets/augmented_data/images datasets/augmented_data/labels datasets/person_dataset
# python datasets/split_dataset.py datasets/augmented_data/cropped/aug_crop_data/images datasets/augmented_data/cropped/aug_crop_data/labels datasets/ppe_dataset
