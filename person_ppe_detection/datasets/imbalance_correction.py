import os
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import numpy as np


def load_class_names(classes_file):
    with open(classes_file, 'r') as file:
        return [line.strip() for line in file.readlines()]


def count_annotations(annotations_dir, class_names):
    class_counts = defaultdict(int)
    for file_name in os.listdir(annotations_dir):
        if file_name.endswith('.txt'):
            with open(os.path.join(annotations_dir, file_name), 'r') as file:
                for line in file:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1

    # Include classes with 0 annotations
    return {class_name: class_counts[id] for id, class_name in enumerate(class_names)}


def plot_class_distribution(class_distribution, title, exclude_zero=False):
    if exclude_zero:
        # Filter out classes with 0 annotations
        class_distribution = {k: v for k, v in class_distribution.items() if v > 0}

    plt.figure(figsize=(11, 7))
    bars = plt.bar(class_distribution.keys(), class_distribution.values(), color=plt.cm.tab10(np.arange(len(class_distribution))))
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.title(title)
    plt.xticks(rotation=35)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom')

    plt.show()


def remove_low_annotation_classes(annotations_dir, class_distribution, threshold, class_names):
    classes_to_remove = {class_name for class_name, count in class_distribution.items() if count <= threshold}

    for file_name in os.listdir(annotations_dir):
        if file_name.endswith('.txt'):
            file_path = os.path.join(annotations_dir, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            with open(file_path, 'w') as file:
                for line in lines:
                    class_id = int(line.split()[0])
                    class_name = class_names[class_id]
                    if class_name not in classes_to_remove:
                        file.write(line)

    return classes_to_remove


def main():
    parser = argparse.ArgumentParser(
        description="Count, plot, and optionally remove low-annotation classes in YOLO annotation files.")
    parser.add_argument('annotations_dir', type=str, help="Directory containing YOLO annotation files")
    parser.add_argument('classes_file', type=str, help="File containing class names")

    args = parser.parse_args()

    # Load class names
    class_names = load_class_names(args.classes_file)

    # Count annotations for each class
    class_distribution = count_annotations(args.annotations_dir, class_names)

    # Plot class distribution including all classes
    plot_class_distribution(class_distribution, title="Class Distribution (All Classes)", exclude_zero=False)

    # Ask user if they want to remove low-annotation classes
    remove_classes = input("Do you want to remove classes with low annotations? (yes/no): ").strip().lower()

    if remove_classes == 'yes':
        threshold = int(
            input("Enter the threshold for removal (classes with annotations <= threshold will be removed): "))
        removed_classes = remove_low_annotation_classes(args.annotations_dir, class_distribution, threshold,
                                                        class_names)
        print(f"Removed annotations for classes with counts <= {threshold}: {', '.join(removed_classes)}")

        # Re-count annotations and plot distribution again
        class_distribution = count_annotations(args.annotations_dir, class_names)
        plot_class_distribution(class_distribution, title="Class Distribution (After removal of low data classes)", exclude_zero=True)
    else:
        print("No classes were removed.")


if __name__ == "__main__":
    main()

# python datasets/imbalance_correction.py datasets/yolo_labels datasets/classes.txt
# python datasets/imbalance_correction.py datasets/augmented_data/all_labels datasets/classes.txt
# python datasets/imbalance_correction.py datasets/augmented_data/cropped/labels datasets/cropped_classes.txt