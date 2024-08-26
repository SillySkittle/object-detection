# script to trim all other labels from the label files and just keep the annotations for "person"
import os
import argparse


def load_classes(class_file):
    with open(class_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def main(label_dir, valid_class_name, classes, output_dir):
    # Create a mapping from class name to index
    class_to_index = {name: str(index) for index, name in enumerate(classes)}

    valid_class_index = class_to_index.get(valid_class_name, None)

    if valid_class_index is None:
        print(f"Class name '{valid_class_name}' not found in classes file.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label_file in os.listdir(label_dir):
        file_path = os.path.join(label_dir, label_file)
        output_path = os.path.join(output_dir, label_file)

        with open(file_path, 'r') as f:
            lines = f.readlines()
        with open(output_path, 'w') as f:
            for line in lines:
                parts = line.strip().split()
                if parts[0] == valid_class_index:
                    # Replace class index with class name
                    #parts[0] = valid_class_name
                    f.write(" ".join(parts) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and update label files.")
    parser.add_argument('label_dir', type=str, help="Directory containing label files.")
    parser.add_argument('valid_class_name', type=str, help="Class name to keep in labels.")
    parser.add_argument('classes_file', type=str, help="Path to file containing class names.")
    parser.add_argument('output_dir', type=str, help="Directory to save updated label files.")

    args = parser.parse_args()

    classes = load_classes(args.classes_file)
    main(args.label_dir, args.valid_class_name, classes, args.output_dir)

# python datasets/label_correction.py datasets/augmented_data/all_labels person datasets/classes.txt datasets/augmented_data/labels