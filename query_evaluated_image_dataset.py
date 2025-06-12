# query_evaluated_image_dataset.py

"""
Query Evaluated Image Dataset Script

This script loads processed parquet files and allows flexible querying of detected faces:

- You can define a query.txt with optional parameters:
    - min_face_conf
    - min_face_width
    - min_face_height
    - include_classes_AND (mutually exclusive with include_classes_OR)
    - include_classes_OR (mutually exclusive with include_classes_AND)
    - exclude_classes_AND

- The script will parse the query.txt, build a dynamic SQL query, and execute it.
- It will output a CSV with one row per matching face, including the image bytes for visualization.
- Optionally, it will visualize the faces if --create_face_visualizations is True.

Usage example:

python query_evaluated_image_dataset.py \
    --input_folder data/processed/ \
    --query_config query.txt \
    --create_face_visualizations True \
    --debug True

Author: Trevor Gribble
Date: 2025-06-11
"""

# Import libraries
import argparse
import os
import sys
import glob
import pandas as pd
import duckdb
import ast
import re
from datetime import datetime

# Import visualize_faces from separate module (you need to implement this!)
from visualize_faces import visualize_faces

# Function to parse query.txt into a QueryConfig dictionary
def parse_query_config(query_config_path):
    config = {
        'min_face_conf': None,
        'min_face_width': None,
        'min_face_height': None,
        'include_classes_AND': [],
        'include_classes_OR': [],
        'exclude_classes_AND': []
    }

    if not os.path.exists(query_config_path):
        print(f"❌ ERROR: query_config file '{query_config_path}' not found.")
        sys.exit(1)

    with open(query_config_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        # Skip comments and blank lines
        if line.startswith("#") or line == "":
            continue

        # Split on first '='
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            if key in ['min_face_conf', 'min_face_width', 'min_face_height']:
                config[key] = float(value) if '.' in value else int(value)
            elif key in ['include_classes_AND', 'include_classes_OR', 'exclude_classes_AND']:
                config[key] = ast.literal_eval(value)

    # Sanity check: enforce that only ONE of include_classes_AND or include_classes_OR is used
    if config['include_classes_AND'] and config['include_classes_OR']:
        print("❌ ERROR: You must use only ONE of include_classes_AND or include_classes_OR — not both.")
        sys.exit(1)

    return config

# Function to build the dynamic SQL query string
def build_query(config, debug):

    # Collect all unique class names required for MAX CASE columns
    all_classes = set()
    for class_name, _ in config['include_classes_AND']:
        all_classes.add(class_name)
    for class_name, _ in config['include_classes_OR']:
        all_classes.add(class_name)
    for class_name, _ in config['exclude_classes_AND']:
        all_classes.add(class_name)

    query_classes = sorted(list(all_classes))

    # Build SQL
    # If no class filters are active, use simpler query without attribute processing
    if not any([config['include_classes_AND'], config['include_classes_OR'], config['exclude_classes_AND']]):
        sql_body = """
        WITH faces_exploded AS (
            SELECT
                row_number() OVER () - 1 AS image_id,
                image_url,
                UNNEST(faces) AS face
            FROM images
            WHERE faces IS NOT NULL
        )
        SELECT DISTINCT
            image_id,
            image_url,
            face.bbox AS bbox,
            face.confidence AS face_conf,
            face.width AS face_width,
            face.height AS face_height,
            NULL AS face_attributes
        FROM faces_exploded
        WHERE TRUE
        """
        if config['min_face_conf'] is not None:
            sql_body += f"\n    AND face.confidence >= {config['min_face_conf']}"
        if config['min_face_width'] is not None:
            sql_body += f"\n    AND face.width >= {config['min_face_width']}"
        if config['min_face_height'] is not None:
            sql_body += f"\n    AND face.height >= {config['min_face_height']}"
        sql_body += "\nORDER BY face.confidence DESC"
        return sql_body, []
        
    # If we have class filters, use the full query with attribute processing
    sql_body = """
    WITH faces_exploded AS (
        SELECT
            row_number() OVER () - 1 AS image_id,
            image_url,
            UNNEST(faces) AS face
        FROM images
        WHERE faces IS NOT NULL
    ),
    attrs_exploded AS (
        SELECT
            image_id,
            image_url,
            face.bbox AS bbox,
            face.confidence AS face_conf,
            face.width AS face_width,
            face.height AS face_height,
            face.face_attributes AS face_attributes,
            UNNEST(CASE 
                WHEN face.face_attributes IS NULL THEN ARRAY[json_object('class_name', NULL, 'confidence', NULL)]
                ELSE face.face_attributes 
            END) AS attr
        FROM faces_exploded
    ),
    attrs_grouped AS (
        SELECT
            image_id,
            image_url,
            bbox,
            face_conf,
            face_width,
            face_height,
            attr.class_name,
            MAX(attr.confidence) as max_conf
        FROM attrs_exploded
        WHERE $classes = array[]::varchar[] OR attr.class_name = ANY (SELECT UNNEST($classes))
        GROUP BY image_id, image_url, bbox, face_conf, face_width, face_height, attr.class_name
    ),
    attrs_filtered AS (
        SELECT
            image_id,
            image_url,
            bbox,
            face_conf,
            face_width,
            face_height,
            ARRAY_AGG(
                json_object(
                    'class_name', class_name,
                    'confidence', max_conf
                )
            ) AS filtered_attributes,
    """

    # Add dynamic MAX CASE columns
    conf_columns = []
    for class_name in all_classes:
        conf_columns.append(
            f"            MAX(CASE WHEN class_name = '{class_name}' THEN max_conf ELSE NULL END) AS {class_name.lower()}_conf"
        )

    sql_body += ",\n".join(conf_columns)

    sql_body += """
        FROM attrs_grouped
        GROUP BY image_id, image_url, bbox, face_conf, face_width, face_height
    )
    SELECT DISTINCT
        image_id,
        image_url,
        bbox,
        face_conf,
        face_width,
        face_height,
        filtered_attributes AS face_attributes
    FROM attrs_filtered
    WHERE TRUE
    """

    # Add filters
    if config['min_face_conf'] is not None:
        sql_body += f"\n    AND face_conf >= {config['min_face_conf']}"

    if config['min_face_width'] is not None:
        sql_body += f"\n    AND face_width >= {config['min_face_width']}"

    if config['min_face_height'] is not None:
        sql_body += f"\n    AND face_height >= {config['min_face_height']}"

    # Include_classes_AND
    for (class_name, threshold) in config['include_classes_AND']:
        sql_body += f"\n    AND {class_name.lower()}_conf >= {threshold}"

    # Include_classes_OR
    if len(config['include_classes_OR']) > 0:
        or_conditions = []
        for (class_name, threshold) in config['include_classes_OR']:
            or_conditions.append(f"{class_name.lower()}_conf >= {threshold}")
        if or_conditions:
            sql_body += "\n    AND (" + " OR ".join(or_conditions) + ")"

    # Exclude_classes_AND
    for class_name, threshold in config['exclude_classes_AND']:
        sql_body += f"\n    AND ({class_name.lower()}_conf IS NULL OR {class_name.lower()}_conf < {threshold})"

    sql_body += "\nORDER BY face_conf DESC"

    if debug:
        print("\n=== Final Generated SQL ===\n")
        print(sql_body)

    return sql_body, query_classes

# Function to build short suffix string for output filename
def build_query_suffix(config):
    parts = []
    if config['min_face_conf'] is not None:
        parts.append(f"conf{config['min_face_conf']}")
    if config['min_face_width'] is not None:
        parts.append(f"w{config['min_face_width']}")
    if config['min_face_height'] is not None:
        parts.append(f"h{config['min_face_height']}")
    if config['include_classes_AND']:
        parts.append("AND_" + "_".join([cls for cls, _ in config['include_classes_AND']]))
    if config['include_classes_OR']:
        parts.append("OR_" + "_".join([cls for cls, _ in config['include_classes_OR']]))
    if config['exclude_classes_AND']:
        parts.append("EX_" + "_".join([cls for cls, _ in config['exclude_classes_AND']]))

    suffix = "_".join(parts)
    # Clean suffix
    suffix = re.sub(r'[^A-Za-z0-9_]', '', suffix)
    return suffix

# Main function
def main():
    parser = argparse.ArgumentParser(description="Query Evaluated Image Dataset")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing processed parquet files.")
    parser.add_argument("--query_config", type=str, required=True, help="Path to query.txt config file.")
    parser.add_argument("--create_face_visualizations", type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to create face visualizations.")
    parser.add_argument("--debug", type=lambda x: (str(x).lower() == 'true'), default=True, help="Enable debug prints.")
    args = parser.parse_args()

    # Load config
    config = parse_query_config(args.query_config)
    if args.debug:
        print("\n✅ Parsed Query Config:")
        for k, v in config.items():
            print(f"  {k} = {v}")

    # Build dynamic output filename
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    suffix = build_query_suffix(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(output_dir, f"query_results_{suffix}_{timestamp}.csv")

    if args.create_face_visualizations:
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

    # Load parquet files
    parquet_files = sorted(glob.glob(os.path.join(args.input_folder, "*.parquet")))

    if len(parquet_files) == 0:
        print(f"\n❌ ERROR: No parquet files found in '{args.input_folder}'.")
        sys.exit(1)

    print(f"\n✅ Loading {len(parquet_files)} parquet files into DuckDB...")

    con = duckdb.connect()
    con.execute("""
        CREATE TABLE images AS 
        SELECT * FROM read_parquet($1)
    """, (parquet_files,))

    total_rows = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    print(f"Total rows in images table: {total_rows}")

    # Build and run query
    query_sql, query_classes = build_query(config, args.debug)

    print("\n✅ Running query...")
    # Only pass parameters if we have class filters
    if any([config['include_classes_AND'], config['include_classes_OR'], config['exclude_classes_AND']]):
        results_df = con.execute(query_sql, {'classes': query_classes}).fetchdf()
    else:
        results_df = con.execute(query_sql).fetchdf()

    print(f"\n✅ Query returned {len(results_df)} faces.")

    # Save CSV
    print(f"\n✅ Saving query results to {output_csv} ...")
    results_df.to_csv(output_csv, index=False)
    print("✅ Done saving CSV.")

    # Optionally visualize
    if args.create_face_visualizations:
        print("\n✅ Creating face visualizations...")
        visualize_faces(
            results_df,
            input_parquet_folder=args.input_folder,
            output_dir=os.path.join(output_dir, "visualizations"),
            debug=args.debug,
            query_suffix=suffix,
            config=config
        )

    # Close DuckDB
    con.close()
    print("\n✅ Done.")

# Entry point
if __name__ == "__main__":
    main()
