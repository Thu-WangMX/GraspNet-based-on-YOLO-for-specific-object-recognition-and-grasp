# labelme_to_yolo_multiclass.py

import os
import json
import shutil
from tqdm import tqdm

def convert_multiclass_json_to_yolo(json_dir, output_dir, class_mapping):
    """
    Converts a directory of LabelMe JSON files (with multiple classes) 
    to YOLO format TXT files.

    Args:
        json_dir (str): Directory containing LabelMe JSON files and their corresponding images.
        output_dir (str): Directory where the output YOLO format TXT files will be saved.
        class_mapping (dict): A dictionary mapping class names (e.g., 'liquid') to class IDs (e.g., 0).
    """
    # Ensure the output directory is clean
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"Cleaned and created output directory: {output_dir}")

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"❌ Error: No .json files found in '{json_dir}'.")
        return

    print(f"Found {len(json_files)} JSON files to convert.")

    for json_filename in tqdm(json_files, desc="Converting Multi-Class JSON to YOLO TXT"):
        json_path = os.path.join(json_dir, json_filename)

        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Warning: Skipping invalid JSON file: {json_filename}")
                continue

        # Get image dimensions from the JSON file
        img_width = data.get('imageWidth')
        img_height = data.get('imageHeight')

        if not img_width or not img_height:
            print(f"⚠️ Warning: imageWidth/imageHeight not found in {json_filename}. Skipping.")
            continue
        
        yolo_annotations = []
        # Iterate through all annotated shapes in the JSON
        for shape in data.get('shapes', []):
            label = shape.get('label')
            
            # Check if the label is one of the classes we care about
            if label not in class_mapping:
                print(f"⚠️ Warning: Label '{label}' in {json_filename} is not in class_mapping. Skipping this shape.")
                continue
            
            class_id = class_mapping[label]
            
            if shape.get('shape_type') != 'rectangle':
                print(f"⚠️ Warning: Shape type is not 'rectangle' in {json_filename}. Skipping this shape.")
                continue

            points = shape.get('points')
            if not points or len(points) != 2:
                print(f"⚠️ Warning: Invalid points for rectangle in {json_filename}. Skipping.")
                continue

            # Extract coordinates and calculate YOLO format values
            x1 = min(points[0][0], points[1][0])
            y1 = min(points[0][1], points[1][1])
            x2 = max(points[0][0], points[1][0])
            y2 = max(points[0][1], points[1][1])

            bbox_width = x2 - x1
            bbox_height = y2 - y1
            center_x = x1 + bbox_width / 2
            center_y = y1 + bbox_height / 2

            # Normalize coordinates
            center_x_norm = center_x / img_width
            center_y_norm = center_y / img_height
            width_norm = bbox_width / img_width
            height_norm = bbox_height / img_height

            yolo_annotations.append(f"{class_id} {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}")

        # Write all annotations for the image to its corresponding .txt file
        if yolo_annotations:
            txt_filename = os.path.splitext(json_filename)[0] + '.txt'
            txt_path = os.path.join(output_dir, txt_filename)
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))

def main():
    """Main function to set up paths and class mappings, then run the conversion."""
    print("--- Starting Multi-Class LabelMe JSON to YOLO TXT Conversion ---")
    
    # --- CRITICAL ---
    # Define the mapping from your string labels to integer class IDs.
    # YOLO requires integer IDs starting from 0.
    CLASS_MAPPING = {
        "liquid": 0,
        "solid": 1
    }
    print(f"Using Class Mapping: {CLASS_MAPPING}")
    
    # Define source and destination directories
    source_dir = '/home/hjj/hd10k_unet_stain_segmentation/mix_yolo_annotation/images'
    output_label_dir = '/home/hjj/hd10k_unet_stain_segmentation/mix_yolo_annotation/labels'
    
    convert_multiclass_json_to_yolo(
        json_dir=source_dir,
        output_dir=output_label_dir,
        class_mapping=CLASS_MAPPING
    )
    
    print(f"\n✅ Conversion successful! YOLO labels saved to: '{output_label_dir}'")

if __name__ == "__main__":
    main()