# predict_single.py

import os
import cv2
import argparse
from ultralytics import YOLO

def predict_single_image(weights_path, source_image_path, output_dir, confidence_threshold):
    """
    Uses a trained YOLOv8 model to perform object detection on a single image and saves the result.

    Args:
        weights_path (str): The path to the trained model's .pt weights file.
        source_image_path (str): The path to the single image file for prediction.
        output_dir (str): The directory where the annotated output image will be saved.
        confidence_threshold (float): The minimum confidence score for a detection to be considered valid.
    """
    print("--- Starting Single Image Prediction ---")
    
    # 1. Validate inputs
    if not os.path.exists(weights_path):
        print(f"❌ Error: Model weights not found at '{weights_path}'")
        return
    if not os.path.exists(source_image_path):
        print(f"❌ Error: Source image not found at '{source_image_path}'")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Results will be saved in: '{output_dir}'")

    # 2. Load the trained YOLO model
    try:
        print(f"🧠 Loading model from '{weights_path}'...")
        model = YOLO(weights_path)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Perform prediction on the source image
    print(f"🖼️ Predicting on image: '{source_image_path}'...")
    try:
        # The `predict` method returns a list of result objects
        results = model.predict(
            source=source_image_path,
            conf=confidence_threshold,
            verbose=False  # Set to True for more detailed model output
        )
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return

    # 4. Process and save the result
    # For a single image, there will be only one result object in the list
    if not results:
        print("⚠️ No results were returned from the model.")
        return
        
    result = results[0]
    
    # The `plot()` method is a convenient way to get the annotated image (with boxes and labels)
    annotated_image = result.plot()
    
    # Construct the output path
    base_filename = os.path.basename(source_image_path)
    output_filename = f"predicted_{base_filename}"
    output_path = os.path.join(output_dir, output_filename)

    # Save the annotated image
    cv2.imwrite(output_path, annotated_image)
    
    # 5. Print summary
    num_detections = len(result.boxes)
    print(f"✅ Prediction complete. Found {num_detections} instances of stains.")
    print(f"💾 Annotated image saved to: '{output_path}'")
    print("-----------------------------------------")


if __name__ == "__main__":
    # Setup command-line argument parser
    parser = argparse.ArgumentParser(description="Perform YOLOv8 prediction on a single image.")
    
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the trained YOLOv8 model weights file (.pt)."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the source image file for prediction."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="single_prediction_output",
        help="Directory to save the prediction result. (default: 'single_prediction_output')"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detection. (default: 0.5)"
    )

    args = parser.parse_args()

    # Call the main prediction function with the provided arguments
    predict_single_image(
        weights_path=args.weights,
        source_image_path=args.source,
        output_dir=args.output_dir,
        confidence_threshold=args.conf
    )