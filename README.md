# Key Finder CNN

A Python machine learning program that takes an audio file in **WAV** or **MP3** format, converts it into a **spectrogram**, and predicts the **musical key** of the song using a **Convolutional Neural Network (CNN)**.

The model supports predicting among **24 key classes** (Note: All class examples will use flat notation):

- 12 major keys
- 12 minor keys

Examples:
- C major
- Eb major
- A minor
- Gb minor

---

## Overview

This project performs **musical key classification** from audio by following this pipeline:

1. Load an audio file
2. Convert the file into a time-frequency representation
   - **Mel spectrogram** by default
   - **CQT spectrogram** optionally
3. Normalize the input data
4. Feed the spectrogram into a CNN
5. Predict one of 24 possible key classes

The program can be used in two main ways:

- **Train a model** on a labeled dataset
- **Predict the key** of a single song using a saved model

---

## Features

- Accepts **.wav** and **.mp3** input through `librosa`
- Uses **Mel spectrograms** by default
- Optional **Constant-Q Transform (CQT)** mode
- CNN-based classifier implemented in **PyTorch**
- Supports:
  - training
  - evaluation
  - single-song prediction
  - batch testing
- Handles **24-key classification**
- Includes **weighted sampling** to help with class imbalance

---

## Model Pipeline

### 1. Audio Input
The program loads audio using `librosa.load()`.

- In mel mode, audio is resampled to **44.1 kHz**
- In CQT mode, default `librosa` loading is used

### 2. Spectrogram Conversion

#### Mel Spectrogram (default)
The input song is converted into a mel spectrogram using:

- `n_fft = 8192`
- `n_mels = 105`
- `hop_length = 8820`

The power spectrogram is then converted to decibel scale using:

```python
librosa.power_to_db(...)