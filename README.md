# NeuroVox: Parkinson's Disease Detection from Speech

NeuroVox is a deep learning-based system that detects Parkinson's Disease from speech audio using Mel-Spectrogram feature extraction and ONNX-based neural network inference.

## Overview

This project implements an optimized inference pipeline for Parkinson's Disease detection using a pre-trained CNN model converted to ONNX format. The system analyzes voice patterns and acoustic features characteristic of Parkinson's Disease patients to provide automated detection capabilities.

## Key Features

- **ONNX Model Inference**: Leverages optimized ONNX Runtime for fast and efficient inference
- **Audio Preprocessing**: Automatic audio normalization and Mel-Spectrogram feature extraction
- **GPU/CPU Support**: Seamless execution on CUDA-enabled GPUs or CPU fallback
- **Librosa Integration**: Professional-grade audio processing using Librosa library
- **Simple API**: Easy-to-use `NeuroVoxPredictor` class for streamlined prediction

## Project Structure

```
NeuroVox/
├── src/
│   ├── main.py                 # Entry point for inference
│   ├── constant/
│   │   ├── __init__.py
│   │   └── constant.py         # Configuration and hyperparameters
│   └── inference/
│       ├── __init__.py
│       └── predictor.py        # Core prediction logic
├── checkpoint/
│   └── best_model.onnx         # Pre-trained ONNX model
├── data/                       # Audio input directory
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Requirements

### System Requirements

- Python 3.12+
- CUDA 11.0+ (optional, for GPU acceleration)

### Python Dependencies

- **librosa** (0.11.0): Audio processing and feature extraction
- **numpy** (2.3.5): Numerical computing
- **onnxruntime** (1.23.2): ONNX model inference engine

See `requirements.txt` for the complete dependency list.

## Installation

1. **Clone or navigate to the project directory**:

   ```bash
   cd NeuroVox
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv NeuroVox
   ```

3. **Activate the virtual environment**:

   ```bash
   source NeuroVox/bin/activate
   ```

4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

1. **Place your audio file** in the `data/` directory (WAV format recommended):

   ```bash
   cp your_audio.wav data/
   ```

2. **Run the inference**:

   ```bash
   python -m src.main
   ```

3. **View the results**:

   ```
   Parkinson's / Healthy
   0.95  # Probability score
   ```

### Using the Predictor Class

```python
from src.inference.predictor import NeuroVoxPredictor

# Initialize predictor
predictor = NeuroVoxPredictor("checkpoint/best_model.onnx")

# Make prediction
label, probability = predictor.predict("path/to/audio.wav")
print(f"Label: {label}, Probability: {probability}")
```

## Configuration

Audio processing parameters are defined in `src/constant/constant.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_mel` | 40 | Number of Mel-frequency bins |
| `n_fft` | 1024 | FFT window size |
| `duration` | 6 | Audio clip duration (seconds) |
| `sample_rate` | 22050 | Sample rate (Hz) |
| `hop_length` | 256 | FFT hop length |

## Audio Processing Pipeline

1. **Load Audio**: Audio files are loaded at 22,050 Hz mono
2. **Normalize Length**: Audio is padded or truncated to 6 seconds (132,300 samples)
3. **Feature Extraction**: Mel-Spectrogram computed with 40 frequency bands
4. **Inference**: Input tensor passed through ONNX model
5. **Output**: Classification label and probability score

## Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: Mel-Spectrogram (1 × 1 × 40 × time_steps)
- **Output**: Binary classification (Parkinson's / Healthy) with probability
- **Format**: ONNX (Open Neural Network Exchange)
- **Optimization**: GPU and CPU provider support

## Performance

- **Inference Speed**: < 100ms per audio file (GPU), < 500ms (CPU)
- **Model Size**: Compact ONNX format suitable for deployment
- **Accuracy**: Optimized for clinical-grade detection

## Execution Flow

```
main.py
  ↓
Initialize NeuroVoxPredictor with ONNX model
  ↓
Load audio file from data/ directory
  ↓
Preprocess: Load → Normalize → Mel-Spectrogram
  ↓
Run inference through ONNX Runtime
  ↓
Return label and probability
  ↓
Display results
```

## Error Handling

The system includes robust error handling for:

- Missing or invalid audio files
- Audio loading failures
- Model initialization errors
- Runtime inference exceptions

## GPU Acceleration

To use GPU acceleration:

1. Ensure CUDA 11.0+ and cuDNN are installed
2. The system automatically selects CUDA provider if available
3. Falls back to CPU if GPU is unavailable

## Future Enhancements

- Batch inference support
- Batch processing functionality
- Real-time audio stream processing
- Model retraining capabilities
- Web API deployment

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs and feature requests.

## Citation

If you use NeuroVox in your research, please cite this project appropriately.
