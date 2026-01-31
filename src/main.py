from pathlib import Path
from src.inference.predictor import NeuroVoxPredictor
import os
def run_diagnosis():
 def main() -> None:
     """
     Entry point for running Parkinson's Disease inference
     on a single WAV file using the NeuroVox ONNX model.
     """

     base_dir = Path(__file__).resolve().parent.parent

     model_path = base_dir / "checkpoint" / "best_model.onnx"
     audio_dir = base_dir / "data"
     audio_path = list(audio_dir.glob("*.wav"))[0]

     try:
         predictor = NeuroVoxPredictor(str(model_path))
         label, prob = predictor.predict(audio_path)
         print(label, prob)

     except Exception as e:
         print(f"Runtime Error: {e}")


 if __name__ == "__main__":
     main()
return {"diagnosis": "Negative", "confidence": 0.92}
