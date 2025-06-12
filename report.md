# WIT Face Detection and Analysis Project Report

## Project Overview
This project implements a scalable pipeline for detecting and analyzing faces in the Wikimedia Image Text (WIT) dataset, with a specific focus on identifying faces with eyeglasses and related attributes.

## Technical Approach

### 1. Face Detection Strategy
- **Primary Detector**: MTCNN (Multi-task Cascaded Convolutional Networks)
  - Chosen for its robustness and ability to handle various face orientations
  - Provides confidence scores and face dimensions needed for filtering
  - Efficiently processes large-scale image datasets

### 2. Attribute Detection
- **Model**: YOLOv8x trained on OpenImagesV7
  - Leverages pre-trained weights for accurate attribute detection
  - Specifically mapped "Glasses" class to "Eyeglasses" to disambiguate from "Sunglasses"
  - Only runs on faces meeting minimum size requirements (60px × 80px) for efficiency

### 3. Data Processing Pipeline
1. **Input Processing**
   - Parquet file handling for efficient storage and querying
   - Streaming approach to manage memory usage
   - Parallel processing capability for large datasets

2. **Face Analysis**
   - Two-stage detection: MTCNN followed by YOLO
   - Confidence thresholding for quality control
   - Attribute aggregation and standardization

3. **Result Storage**
   - Structured parquet format for processed results
   - Maintains original image metadata
   - Optimized for fast querying
   - Processed dataset available on HuggingFace: https://huggingface.co/datasets/bbtre/wit_faces_evaluations/tree/main

### 4. Query System Design
- **Flexible Query Interface**
  - Supports both simple and complex attribute combinations
  - AND/OR logic for class filtering
  - Exclusion criteria support
  
- **DuckDB Integration**
  - Fast in-memory SQL processing
  - Dynamic query generation
  - Efficient handling of large result sets

### 5. Visualization System
- **Modular Design**
  - Separate visualization module for maintainability
  - Configurable grid layouts and pagination
  - Automatic caption generation

### 6. Environment Management Strategy
- **Dual Environment Approach**
  - Separate CPU and GPU environments to prevent dependency conflicts
  - Explicitly versioned dependencies to ensure reproducibility
  - Comprehensive testing of both installation paths

- **Installation Verification**
  - Automated verification steps for both CPU and GPU setups
  - Clear error messages and troubleshooting guides
  - Fallback options for different CUDA versions

- **Documentation Philosophy**
  - Step-by-step installation instructions with verification points
  - Common pitfalls and solutions documented
  - Examples of expected output for validation

### 7. Quality Assurance
- **Environment Testing**
  - Tested on multiple platforms (Linux with/without GPU)
  - Verified compatibility across different CUDA versions
  - Validated all dependencies work together without conflicts

- **Installation Validation**
  - Automated checks for required dependencies
  - Clear feedback on missing or incompatible components
  - Graceful fallbacks for different hardware configurations

## Technical Decisions and Trade-offs

### Key Decisions
1. **Environment Management**
   - Pros: Better reproducibility, clear hardware-specific paths
   - Cons: Maintenance of multiple environment files
   - Decision: Split into CPU/GPU environments to prevent dependency conflicts

2. **Using MTCNN + YOLO vs Single Model**
   - Pros: Better accuracy, separate confidence scores
   - Cons: Additional computational overhead
   - Decision: Benefits of accuracy outweigh performance cost

3. **Parquet + DuckDB vs Traditional DB**
   - Pros: Better compression, columnar storage, faster queries
   - Cons: Requires data preprocessing
   - Decision: Performance advantages justify preprocessing step

4. **Two-Stage Processing**
   - Pros: Memory efficient, allows parallel processing
   - Cons: Longer total processing time
   - Decision: Scalability more important than processing speed

### Alternative Approaches
- Attempted custom classifier using CelebA dataset
- Considered single-pass processing (rejected due to memory constraints)

## Performance

- ~1000 images/minute processing on GPU
- Sub-second query response times
- Efficient memory usage through streaming
- Parallel processing support with configurable batch sizes

## Future Work

1. Batch processing for YOLO inference
2. Additional attribute detection
3. Interactive query interface
4. Distributed processing support

## Conclusions
The system successfully meets requirements for face detection and attribute analysis in the WIT dataset. The chosen technologies (MTCNN, YOLO, DuckDB) provide a good balance of accuracy and performance, while the modular design allows for future improvements. Separate CPU/GPU environments ensure reliable deployment across different hardware configurations.
