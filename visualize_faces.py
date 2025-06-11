# visualize_faces.py

"""
Face Visualization Utility

This script visualizes faces extracted and filtered from image datasets using a tabular query workflow.
Intended for interactive exploration, dataset QA, and presentation of query results.

Key Features:
- Loads image metadata and face region/attribute info from a DataFrame (typically output from DuckDB/parquet).
- For each matched face, crops and displays the region-of-interest from the original image.
- Saves visualizations as paginated PNG grids, with captions summarizing detection confidence, face size, and all detected classes.
- Robustly parses face attribute formats from a variety of pandas/SQL/csv sources.
- Output filenames always match the original query (using the query suffix).

Recommended Workflow:
1. Run a query to filter faces (see `query_evaluated_image_dataset.py`).
2. Use this script to create interpretable, human-friendly visualization grids for downstream QA, presentations, or annotation.

Arguments/Parameters:
- df: DataFrame containing one row per face (with bbox, image_url, face_attributes, etc.).
- input_parquet_folder: Directory containing the original parquet files (with full image data).
- output_dir: Directory to save output visualization grids.
- config: (Optional) Query config dictionary; used for descriptive plot titles.
- query_suffix: Short, unique string representing the query—used in output filenames.
- page_size, faces_per_row: Control grid layout for visualization.
- debug: Print detailed error/info messages.

Example usage:
    visualize_faces(
        results_df,
        input_parquet_folder="data/processed/",
        output_dir="outputs/visualizations",
        config=config,
        debug=True,
        page_size=100,
        faces_per_row=10,
        query_suffix="conf098_w100_h100_OR_Eyeglasses_Hat_EX_Sunglasses"
    )

Typical output:
- outputs/visualizations/faces_conf098_w100_h100_OR_Eyeglasses_Hat_EX_Sunglasses_page_001.png
- ... (additional pages as needed)

Author: Trevor Gribble
Date: 2025-06-11
"""

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import io
import os
import json
import duckdb
import numpy as np

def build_query_description(config):
    """
    Human-readable description of the current query settings.
    Used as a figure title, not for file naming.
    """
    desc = []
    if config.get('min_face_conf') is not None:
        desc.append(f"Conf: {config['min_face_conf']}")
    if config.get('min_face_width') is not None:
        desc.append(f"Min W: {config['min_face_width']}")
    if config.get('min_face_height') is not None:
        desc.append(f"Min H: {config['min_face_height']}")
    if config.get('include_classes_AND'):
        desc.append("AND: " + ", ".join([f"{cls}:{thresh}" for cls, thresh in config['include_classes_AND']]))
    if config.get('include_classes_OR'):
        desc.append("OR: " + ", ".join([f"{cls}:{thresh}" for cls, thresh in config['include_classes_OR']]))
    if config.get('exclude_classes_AND'):
        desc.append("EX: " + ", ".join([f"{cls}:{thresh}" for cls, thresh in config['exclude_classes_AND']]))
    return " | ".join(desc)

def visualize_faces(
    df,
    input_parquet_folder,
    output_dir,
    config=None,
    debug=True,
    page_size=100,
    faces_per_row=10,
    query_suffix=""
):
    """
    Visualize cropped face images based on query results, saving images in a paginated grid.
    
    Args:
        df (pd.DataFrame): DataFrame with one row per face (from DuckDB query).
        input_parquet_folder (str): Path to folder with original parquet files.
        output_dir (str): Where to save visualization images.
        config (dict, optional): Query config, used for readable plot title. Not required for filenames.
        debug (bool): Print verbose debugging info.
        page_size (int): Number of faces per output image page.
        faces_per_row (int): Number of faces per row in each image.
        query_suffix (str): Unique query identifier for output filenames (matches the CSV).
    """
    print(f"✅ Visualizing {len(df)} faces...")

    # 1. Index image bytes from original parquet using DuckDB for fast lookup
    print(f"🔍 Indexing images from: {input_parquet_folder}")
    parquet_files = [
        os.path.join(input_parquet_folder, f)
        for f in os.listdir(input_parquet_folder)
        if f.endswith('.parquet')
    ]
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE images AS 
        SELECT image_url, image['bytes'] as image_bytes 
        FROM read_parquet($1)
    """, (parquet_files,))

    os.makedirs(output_dir, exist_ok=True)
    total_faces = len(df)
    num_pages = (total_faces + page_size - 1) // page_size
    print(f"✅ Preparing {num_pages} pages of {page_size} faces per page...")

    # Use query description for plot title, but query_suffix for filename
    page_title = build_query_description(config) if config else query_suffix

    for page_num in range(num_pages):
        start_idx = page_num * page_size
        end_idx = min(start_idx + page_size, total_faces)
        faces_to_plot = df.iloc[start_idx:end_idx]

        print(f"➡️  Plotting page {page_num+1}/{num_pages} (faces {start_idx+1} to {end_idx})")
        num_faces_on_page = len(faces_to_plot)
        num_rows = (num_faces_on_page + faces_per_row - 1) // faces_per_row
        fig, axes = plt.subplots(num_rows, faces_per_row, figsize=(faces_per_row * 2, num_rows * 2))
        axes = axes.flatten()

        fig.suptitle(f"Query: {page_title} — Page {page_num+1}/{num_pages}", fontsize=14)

        for i, (idx, row) in enumerate(faces_to_plot.iterrows()):
            try:
                # Lookup image bytes from original parquet
                img_bytes = con.execute(
                    "SELECT image_bytes FROM images WHERE image_url = $1 LIMIT 1",
                    [row['image_url']]
                ).fetchone()
                if img_bytes is None:
                    raise Exception("Image bytes not found for this image_url")
                img_bytes = img_bytes[0]
                
                # Handle bbox coordinates
                bbox = row['bbox']
                if isinstance(bbox, str):
                    try:
                        bbox_coords = json.loads(bbox)
                    except Exception:
                        bbox_str = bbox.strip('[]')
                        bbox_coords = [float(x.strip()) for x in bbox_str.split(',')]
                else:
                    bbox_coords = [float(x) for x in bbox]
                if len(bbox_coords) != 4:
                    raise Exception(f"Expected 4 values for bbox, got: {bbox_coords}")
                x1, y1, x2, y2 = map(int, bbox_coords)

                # Load and crop image
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                face_crop = img.crop((x1, y1, x2, y2))
                axes[i].imshow(face_crop)
                axes[i].axis('off')

                # --- Robust face_attributes parsing (WORKING) ---
                face_attrs = row['face_attributes']

                # If it's a numpy array (or list) of strings, flatten and parse as JSON
                if isinstance(face_attrs, np.ndarray) or (
                    isinstance(face_attrs, list) and face_attrs and isinstance(face_attrs[0], str)
                ):
                    flat_list = list(face_attrs)
                    joined = "".join(flat_list)
                    try:
                        face_attrs = json.loads(joined)
                    except Exception as e:
                        if debug:
                            print(f"⚠️ JSON decode error (array/list): {e} on value: {joined}")
                        face_attrs = []
                # If it's just a single string, parse JSON
                elif isinstance(face_attrs, str):
                    try:
                        face_attrs = json.loads(face_attrs)
                    except Exception as e:
                        if debug:
                            print(f"⚠️ JSON decode error (string): {e} on value: {face_attrs}")
                        face_attrs = []
                elif face_attrs is None:
                    face_attrs = []
                # If already a list of dicts, keep as is
                elif isinstance(face_attrs, list):
                    pass
                else:
                    if debug:
                        print(f"⚠️ Unexpected face_attributes type: {type(face_attrs)}; value: {face_attrs}")
                    face_attrs = []

                # **CRITICAL: Wrap dict as single-item list AFTER all above parsing**
                if isinstance(face_attrs, dict):
                    face_attrs = [face_attrs]  # wrap as single-item list
                
                # Build class/conf string
                class_lines = []
                if isinstance(face_attrs, list):
                    for attr in face_attrs:
                        if not (isinstance(attr, dict) and 'class_name' in attr and 'confidence' in attr):
                            if debug:
                                print(f"⚠️ Odd face_attr: {repr(attr)}")
                        cname = attr.get('class_name', '?') if isinstance(attr, dict) else '?'
                        cconf = attr.get('confidence', 0) if isinstance(attr, dict) else 0
                        if cname != '?' and cconf is not None:
                            class_lines.append(f"{cname}:{cconf:.2f}")
                if not class_lines and debug:
                    print(f"⚠️ No classes found for this face. Raw face_attrs: {repr(face_attrs)}")
                
                classes_str = ", ".join(class_lines) if class_lines else "No classes"

                # Compose caption
                caption = (
                    f"Face {i+1}\n"
                    f"H:{row['face_height']} W:{row['face_width']} Conf:{row['face_conf']:.2f}\n"
                    f"{classes_str}"
                )
                axes[i].set_title(caption, fontsize=8)

            except Exception as e:
                print(f"⚠️ Error processing face {i}: {e}")
                print(f"Row data:\n{row}\n")
                axes[i].axis('off')

        # Hide any unused subplot axes
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"faces_{query_suffix}_page_{page_num+1:03d}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved page {page_num+1} to: {output_path}")

    print("✅ Done paginating all pages.")
    con.close()
