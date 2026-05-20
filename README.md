# Pneumonia Detection Using CNN and VGG16

This is a deep learning coursework project for classifying chest X-ray images as Normal, Bacterial Pneumonia, or Viral Pneumonia. The original repository contains notebooks for class distribution analysis, image augmentation, data splitting, CNN/VGG16 model building, and LIME interpretability. This cleaned version keeps those notebooks and adds production-style Python modules for training, inference, Grad-CAM, and a Dash-based model review tool.

Live demo presented with the group project: [Hugging Face Space](https://yokeshwar-p-d.hf.space/)

## Project Goals

- Build a CNN/transfer-learning classifier for chest X-ray images.
- Compare normal, bacterial, and viral pneumonia patterns.
- Use augmentation and train/validation/test splitting for model robustness.
- Add explainability workflows with LIME and Grad-CAM-style heatmaps.
- Present sample predictions in a lightweight Dash review interface.

## Repository Contents

- `src/train_cnn.py` - VGG16 transfer-learning training script.
- `src/inference.py` - reusable model inference helper and CLI.
- `src/gradcam.py` - Grad-CAM heatmap utility.
- `dash_app.py` - Dash review app for sample images and probability display.
- `Final_Code_Part1.ipynb` - original model-building notebook.
- `Class Distribution.ipynb` - original class-balance exploration.
- `Image Processing/` - original augmentation notebooks.
- `Data_Splitting/` - original split workflow.
- `Model_Interpretability_LIME.ipynb` - original LIME interpretability workflow.
- `sample_images/` - small public sample images for interface testing.
- `Presentation.pptx` - final group presentation covering dataset, VGG16 architecture, LIME interpretability, Dash UI, and demo link.

## Group Project Context

This was a group project. The final presentation lists the contributors as Yokeshwar Boopathy, Snega Murugan, Anand Vinoy, Aravindsamy, Muthu Laxman, Abhey Gill Singh, and Sangavi Sambathkumar. The public portfolio version highlights the ML workflow, app interface, and explainability components.

## Dataset

The full chest X-ray dataset is not committed because medical image datasets can be large and licensing-sensitive. Place the dataset locally using this structure:

```text
data/chest_xray/
  train/
  val/
  test/
```

## Run Training

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python src/train_cnn.py
```

The trained model is written to `models/pneumonia_vgg16.keras`, which is ignored by Git.

## Run Inference

```powershell
python src/inference.py --model models/pneumonia_vgg16.keras --image sample_images/person38_virus_83.jpeg.jpg
```

## Run Dash Review App

```powershell
python dash_app.py
```

## Medical Disclaimer

This is an educational ML project, not medical diagnosis software. It should not be used for clinical decisions.
