# STFT & Mel Spectrogram Parameters
n_mel = 40
n_fft = 1024
duration = 6
sample_rate = 22050
hop_length = n_fft // 4

# ONNX Runtime Providers
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
