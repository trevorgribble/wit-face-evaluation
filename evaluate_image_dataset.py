# evaluate_image_dataset.py

"""
Image Dataset Evaluation Script

This script processes a folder of parquet image files:

- Runs MTCNN face detection on each image.
- Runs YOLO object detection on each detected face (if thresholds met).
- Outputs processed parquet files with face metadata and face attributes.

Supports batch processing via --start_index and --parquet_file_count.
Includes DEBUG flag for verbose iterative printing.

Usage example:

python evaluate_image_dataset.py \
    --input_folder data/raw/ \
    --output_folder data/processed/ \
    --start_index 0 \
    --parquet_file_count 2 \
    --debug True


Author: Trevor Gribble
Date: 2025-06-11
"""

# Import libraries for argument parsing, file IO, image processing, ML models
import argparse
import os
import sys
import glob
import pandas as pd
import numpy as np
import io
from PIL import Image
import torch
from facenet_pytorch import MTCNN
from ultralytics import YOLO

# -----------------------------------------------
# Argument Parsing
# -----------------------------------------------

# Define and parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate image dataset with MTCNN and YOLO detections.")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing input parquet files.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save processed parquet files.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index of input files to process (inclusive).")
    parser.add_argument("--parquet_file_count", type=int, default=0, help="Number of parquet files to process.")
    parser.add_argument("--debug", type=lambda x: (str(x).lower() == 'true'), default=True, help="Enable verbose debugging output (default=True).")
    return parser.parse_args()

# -----------------------------------------------
# Main Processing Function
# -----------------------------------------------

def main():
    # Parse input arguments
    args = parse_args()

    # Assign argument values to variables
    input_folder = args.input_folder
    output_folder = args.output_folder
    start_index = args.start_index
    parquet_file_count = args.parquet_file_count
    DEBUG = args.debug

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # -----------------------------------------------
    # Step 1: Scan and validate input parquet files
    # -----------------------------------------------

    # Get sorted list of parquet files in input folder
    parquet_files = sorted(glob.glob(os.path.join(input_folder, "*.parquet")))

    # If no parquet files found, exit with error
    if len(parquet_files) == 0:
        print(f"\n❌ ERROR: No parquet files found in '{input_folder}'.")
        sys.exit(1)

    # Get total number of parquet files found
    total_files = len(parquet_files)

    # If start index is out of range, exit with error
    if start_index >= total_files:
        print(f"\n❌ ERROR: start_index={start_index} is out of range.")
        print(f"Only {total_files} parquet files available.")
        print("Available files:")
        for i, fname in enumerate(parquet_files):
            print(f"  [{i}] {os.path.basename(fname)}")
        sys.exit(1)

    # Compute end index
    end_index = start_index + parquet_file_count

    # If requested range exceeds available files, exit with error
    if end_index > total_files:
        print(f"\n❌ ERROR: Requested start_index={start_index} + parquet_file_count={parquet_file_count} exceeds available files.")
        print(f"Only {total_files} parquet files available.")
        print("Available files:")
        for i, fname in enumerate(parquet_files):
            print(f"  [{i}] {os.path.basename(fname)}")
        sys.exit(1)

    # Select files to process based on start_index and count
    selected_files = parquet_files[start_index:end_index]

    # Print selected files
    print(f"\n✅ Selected {len(selected_files)} parquet files to process:")
    for i, fname in enumerate(selected_files):
        print(f"  [{start_index + i}] {os.path.basename(fname)}")

    # -----------------------------------------------
    # Step 2: Initialize Models (MTCNN + YOLO)
    # -----------------------------------------------

    # Define thresholds for sending faces to YOLO
    YOLO_FACE_MIN_CONFIDENCE = 0.98
    YOLO_FACE_MIN_HEIGHT = 80
    YOLO_FACE_MIN_WIDTH = 60

    # Path to YOLO model weights
    YOLO_MODEL_PATH = "models/yolov8x-oiv7.pt"  # adjust if needed

    # Select device (GPU or CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✅ Using device: {device}")

    # Initialize MTCNN face detector
    print("\n✅ Initializing MTCNN...")
    mtcnn = MTCNN(keep_all=True, device=device)

    # Load YOLO object detector
    print(f"✅ Loading YOLO model from {YOLO_MODEL_PATH}...")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    # -----------------------------------------------
    # Step 3: Process each selected parquet file
    # -----------------------------------------------

    for file_idx, INPUT_PARQUET_PATH in enumerate(selected_files):

        # Print which file is being processed
        print(f"\n==== Processing file: {INPUT_PARQUET_PATH} ====")

        # Load parquet file into dataframe
        df = pd.read_parquet(INPUT_PARQUET_PATH)
        NUM_IMAGES_TO_PROCESS = len(df)

        # Print number of images and dataframe columns
        print(f"Loaded {NUM_IMAGES_TO_PROCESS} images. Columns: {df.columns.tolist()}")

        # Initialize list to hold face data for each image
        faces_column = []

        # Process each image in dataframe
        for idx, row in df.iterrows():
            # Load image bytes
            image_bytes = row['image']['bytes']

            # Convert image bytes to PIL Image
            try:
                img = Image.open(io.BytesIO(image_bytes))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            except Exception as e:
                if DEBUG:
                    print(f"Skipping image {idx} due to load error: {e}")
                faces_column.append([])
                continue

            # Run MTCNN face detection
            try:
                boxes, probs = mtcnn.detect(img)
            except Exception as e:
                if DEBUG:
                    print(f"Skipping image {idx} due to MTCNN error: {e}")
                faces_column.append([])
                continue

            # Initialize list to hold face data for this image
            faces_for_this_image = []

            # If faces detected, process each face
            if boxes is not None and len(boxes) > 0:
                for box, prob in zip(boxes, probs):
                    # Extract bounding box coordinates and size
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1

                    # Initialize face attributes list
                    face_attributes = []

                    # Only send face to YOLO if it meets thresholds
                    if prob >= YOLO_FACE_MIN_CONFIDENCE and w >= YOLO_FACE_MIN_WIDTH and h >= YOLO_FACE_MIN_HEIGHT:
                        # Crop face from image
                        face_crop = img.crop((x1, y1, x2, y2))
                        try:
                            # Run YOLO object detection on face crop
                            results = yolo_model.predict(
                                source=face_crop,
                                device=0 if device == 'cuda' else 'cpu',
                                verbose=False
                            )
                            result = results[0]

                            # If detections found, extract them
                            if result.boxes is not None and len(result.boxes) > 0:
                                class_ids = result.boxes.cls.cpu().numpy()
                                scores = result.boxes.conf.cpu().numpy()
                                class_names = [yolo_model.names[int(cls_id)] for cls_id in class_ids]

                                # Add each detection to face_attributes
                                for cls_name, conf in zip(class_names, scores):
                                    # Map "Glasses" → "Eyeglasses" for Stability.ai required use case
                                    if cls_name == "Glasses":
                                        cls_name = "Eyeglasses"

                                    face_attributes.append({
                                        'class_name': cls_name,
                                        'confidence': float(conf)
                                    })

                            # Print detections if DEBUG is enabled
                            if DEBUG:
                                print(f"Image {idx} face passed thresholds → Face Attributes: {face_attributes}")

                        except Exception as e:
                            if DEBUG:
                                print(f"Skipping YOLO detection on face (image {idx}) due to error: {e}")
                            face_attributes = []

                    else:
                        # If face does not meet thresholds, skip YOLO
                        if DEBUG:
                            print(f"Image {idx} face skipped YOLO check (conf={prob:.3f}, size={int(w)}x{int(h)})")

                    # Build face data dictionary
                    face_data = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(prob),
                        'width': int(w),
                        'height': int(h),
                        'face_attributes': face_attributes
                    }

                    # Add face data to list for this image
                    faces_for_this_image.append(face_data)

            # Add face data list to overall faces_column
            faces_column.append(faces_for_this_image)

            # Print progress every 100 images or if DEBUG enabled
            if (idx + 1) % 100 == 0 or DEBUG:
                print(f"Processed {idx + 1}/{NUM_IMAGES_TO_PROCESS} images...")

        # Save results to dataframe
        df['faces'] = faces_column

        # Build output filename using new convention
        output_file = os.path.join(
            output_folder,
            f"wit_face_eval_{start_index + file_idx:05d}.parquet"
        )

        # Save dataframe to parquet
        print(f"\nSaving processed parquet to {output_file} ...")
        df.to_parquet(output_file)
        print("✅ Done.")

    # All files processed
    print("\n✅ All files processed.")

# -----------------------------------------------
# Entry Point
# -----------------------------------------------

if __name__ == "__main__":
    main()
