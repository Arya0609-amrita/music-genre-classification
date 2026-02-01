# music-genre-classification (MCA Minor Project)
A simple, explainable ML pipeline that classifies short music clips into genres using classic audio features (MFCC, Chroma, Spectral Contrast) and baseline models.

 

## Contents

- notebooks/MusicGenre_Extraction_Training.ipynb` – feature extraction + training

- notebooks/MusicGenre_Demo.ipynb` – (optional) load saved scaler/model and predict a clip

- artifacts/features.csv – extracted features (39 per sample)

- artifacts/scaler.joblib, artifacts/model_LogReg.joblib – best model & scaler

- report/ – final report PDF and PPT (uploaded at submission)

 

> Audio files are **not included** to avoid copyright/size issues. 

> For testing, use short clips from the YouTube Audio Library.

 

## Quick Start

1. Open the demo notebook (or use the training notebook’s last cell).

2. Upload a short audio clip (10–30 s).

3. Run the prediction cell → prints the predicted genre.

 

## Model & Features

- Features: 20 MFCC + 12 Chroma + 7 Spectral Contrast = **39 features/sample**

- Models tried: Logistic Regression, Linear SVM, Random Forest

- Best (on my run): Logistic Regression (Accuracy ≈ 0.889, Macro‑F1 ≈ 0.867)

 

## Environment

Python 3.x, librosa, scikit-learn, numpy, pandas, matplotlib, seaborn.

 

## Note

Results can vary slightly across runs due to small dataset and segmentation.
