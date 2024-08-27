import os
import xml.etree.ElementTree as ET
import argparse


def convert_voc_to_yolo(input_dir, output_dir):
    classes_path = os.path.join(input_dir, "classes.txt") # path to classes file

    with open(classes_path, "r") as f:
        class_names = f.read().strip().splitlines()

    os.makedirs(output_dir, exist_ok=True)

    labels_dir = os.path.join(input_dir, "labels")     # Path to labels

    for label_file in os.listdir(labels_dir):
        if label_file.endswith(".xml"):
            tree = ET.parse(os.path.join(labels_dir, label_file))
            root = tree.getroot()

            size = root.find("size") # get image dimensions
            width = int(size.find("width").text)
            height = int(size.find("height").text)

            yolo_annotations = []

            for obj in root.iter("object"):
                class_name = obj.find("name").text
                if class_name not in class_names:
                    continue

                class_id = class_names.index(class_name)
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)

                # convert to YOLO format
                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height

                yolo_annotations.append(f"{class_id} {x_center} {y_center} {box_width} {box_height}")

            output_file = os.path.join(output_dir, label_file.replace(".xml", ".txt"))  # Write the YOLO annotations to the output file
            with open(output_file, "w") as f:
                f.write("\n".join(yolo_annotations))


def main():
    parser = argparse.ArgumentParser(description="Convert PascalVOC annotations to YOLOv8 format.")
    parser.add_argument("input_dir", type=str, help="Path of input directory containing images and labels.")
    parser.add_argument("output_dir", type=str, help="Path of output directory to save YOLOv8 annotations.")
    args = parser.parse_args()

    convert_voc_to_yolo(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

# python pascalVOC_to_yolo.py datasets yolo_labels