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

### Considered Alternatives
1. **CelebA Dataset Training**
   - Attempted training custom eyeglasses classifier
   - Results weren't competitive with YOLO pre-trained model
   - Dataset bias issues encountered

2. **Single-Pass Processing**
   - Would be faster but memory intensive
   - Problematic for large-scale deployment

## Performance and Scalability

### Current Performance
- Processes approximately 1000 images/minute on GPU
- Query response times < 1s for most operations
- Efficient memory usage through streaming

### Scalability Features
- Parallel processing support
- Batch size configuration
- Memory-efficient data handling
- Indexed query operations

## Future Improvements

### Short Term
1. Implement batch processing for YOLO inference
2. Add more attribute classes from OpenImagesV7
3. Optimize memory usage in visualization module

### Long Term
1. Develop custom model for fine-grained attribute detection
2. Implement distributed processing support
3. Add interactive query interface
4. Explore additional datasets for training

## Lessons Learned

### Environment Management
- Importance of clear, tested installation instructions
- Value of separating CPU and GPU environments
- Need for explicit version pinning in dependencies
- Benefits of automated environment validation

### Development Practices
- Importance of testing on different hardware configurations
- Value of comprehensive error handling
- Benefits of modular design for maintainability
- Need for clear documentation and examples

## Conclusions
The implemented system successfully meets the initial requirements while providing a flexible foundation for future extensions. Special attention was paid to environment management and installation processes, ensuring that the system can be reliably deployed across different hardware configurations. The modular design allows for easy updates and improvements, while the chosen technologies (MTCNN, YOLO, DuckDB) provide a good balance of accuracy, speed, and scalability.

A key success factor was the decision to separate CPU and GPU environments, which significantly improved reliability and reduced setup issues. The comprehensive documentation and validation steps ensure that users can successfully deploy the system regardless of their hardware capabilities.
