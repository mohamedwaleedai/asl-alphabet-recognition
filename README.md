# asl-alphabet-recognition

Real-time recognition of the 26 ASL fingerspelling letters from a webcam, using MediaPipe hand landmarks, 121 geometric features and a small Keras MLP.

## Problem

Fingerspelling is how names and unfamiliar words are signed in American Sign Language. This project recognises the letters A-Z from a live camera feed and lets the user build words from them, as a lightweight alternative to image-based classifiers that need a large CNN and are sensitive to background and lighting.

## Approach

```
webcam frame -> MediaPipe HandLandmarker -> 21 landmarks -> 121 features -> MLP -> 5-frame vote -> letter
```

- **Landmarks.** MediaPipe's `hand_landmarker.task` model detects one hand and returns 21 3D landmarks. The classifier never sees pixels.
- **Features** (`src/feature_extractor.py`). 121 values computed from the landmarks: normalised x/y coordinates, z-depth, fingertip-to-palm and fingertip-to-wrist distances, inter-fingertip distances, finger extension ratios, finger curl angles, thumb-finger distances, finger heights, spread angles, palm orientation, aspect ratio and thumb direction.
- **Model** (`train.py`). An MLP with three hidden blocks of 256, 128 and 64 units. Each block is Dense (L2 1e-4), BatchNormalization, LeakyReLU (0.1) and Dropout (0.3). The output is a 26-way softmax. 75,866 parameters.
- **Training.** Up to 1,000 images per letter are converted to landmark features once. The data is split 85/15 (stratified, seed 42). The training set is tripled with Gaussian noise (sigma 0.02), and training uses balanced class weights, label smoothing 0.1, Adam (1e-3), batch size 128, up to 100 epochs, early stopping (patience 20 on validation accuracy) and learning-rate reduction on plateau. The best checkpoint by validation accuracy is saved.
- **Inference** (`app.py`). Predictions from the last 5 frames are majority-voted. A letter is accepted only if its average confidence passes a threshold (0.5 by default, 0.6 for A, S, T, E, U, V, R and 0.65 for M, N). A letter held for 12 consecutive frames at confidence 0.6 or higher is appended to the current word.
- **Interface.** A Tkinter desktop app with the camera view, the current letter and confidence, a word buffer, and space, backspace and clear controls.

## Results

- The training run that produced the shipped model recorded **98.88% validation accuracy**. This number is not reproducible from the contents of this repository: the training images and training logs are not included.
- That figure comes from the same validation split that was used for early stopping and checkpoint selection, so it is an optimistic estimate. There is no held-out test set and no evaluation on live webcam input.
- J and Z are signed with motion in ASL. The model classifies single frames, so it treats them as static poses.
- Measured on a CPU with TensorFlow 2.21 on Windows: the model file is 949 KB, and one single-sample forward pass takes about 6.6 ms. End-to-end frame rate was not measured.

## Tech Stack

- Python 3.11
- TensorFlow / Keras (classifier)
- MediaPipe Tasks (hand landmarks)
- OpenCV and Pillow (capture and display)
- scikit-learn (split, class weights, report)
- Tkinter (GUI)

## Getting Started

Requires Python 3.11 with Tkinter and a webcam.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

`app.py` loads `models/asl_landmark_model.h5` and `src/hand_landmarker.task`, both included in the repository. The GUI and camera loop were not exercised in the environment used to prepare this repository (no webcam); the model loading, feature extraction and predictor were.

To retrain, point the script at a folder that has one sub-folder of images per letter (`A/`, `B/`, ... `Z/`). The layout matches the Kaggle "ASL Alphabet" training set; other folders in that set are ignored.

```bash
python train.py --dataset path/to/asl_alphabet_train
# or: set ASL_DATASET_PATH=path/to/asl_alphabet_train
```

Training overwrites `models/asl_landmark_model.h5`.

## Project Structure

```
app.py                       Tkinter webcam application and predictor
train.py                     landmark extraction and model training
src/
  feature_extractor.py       121 geometric features from 21 landmarks
  hand_landmarker.task       MediaPipe hand landmark model
models/
  asl_landmark_model.h5      trained MLP
requirements.txt
```

## License

MIT. See [LICENSE](LICENSE).

## Status

Complete as a working demo. Accuracy has not been independently verified, and there is no test suite.
