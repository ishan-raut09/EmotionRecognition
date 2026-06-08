# Sentira Emotion Recognition

Sentira is a Streamlit application for facial emotion recognition from uploaded images or a live webcam stream. It detects faces with OpenCV Haar cascades and classifies expressions with a PyTorch ResNet18 model trained for the FER2013 emotion classes.

## Emotion Classes

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Project Structure

```text
app.py                 Streamlit application
src/model.py           ResNet18 and legacy VGG model definitions
src/utils.py           Face detection and preprocessing helpers
src/train.py           FER2013 training script
models/emotion_model.pth
data/fer2013.csv
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the App

```powershell
streamlit run app.py
```

The app supports two webcam modes:

- Cloud WebRTC: works better for deployed Streamlit apps.
- Local OpenCV: useful when running on your own machine.

If `models/emotion_model.pth` is unavailable or cannot be loaded, the app switches to demo mode and still performs face detection without emotion predictions.

## Train the Model

Place the FER2013 CSV at `data/fer2013.csv`, then run:

```powershell
python -m src.train
```

The best model is saved to `models/emotion_model.pth`.

The training script now uses the official FER2013 split: `Training` for fitting, `PublicTest` for validation, and `PrivateTest` for final testing. It also uses balanced sampling, focal loss, stronger augmentation, AdamW fine-tuning, mixed precision on CUDA, and horizontal-flip test-time augmentation.

Useful options:

```powershell
python -m src.train --image-size 224 --batch-size 64 --epochs 40 --patience 8
```

If training is too slow on CPU, reduce the image size:

```powershell
python -m src.train --image-size 96 --batch-size 64
```

For best accuracy, keep pretrained ImageNet weights enabled. If your environment cannot download or use cached torchvision weights, run:

```powershell
python -m src.train --no-pretrained
```
