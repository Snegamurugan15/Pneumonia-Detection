# Pneumonia Detection

Coursework project exploring pneumonia detection from chest X-ray images with deep learning. The repository contains notebooks for image preprocessing, augmentation, data splitting, CNN/VGG-style model work, class distribution analysis, and LIME-based model interpretability.

## Project Context

This is a group coursework project, maintained here under Snega Murugan's GitHub profile as a clean portfolio copy. Original collaboration repository: [Sangavisambathkumar/Pneumonia-Detection](https://github.com/Sangavisambathkumar/Pneumonia-Detection).

## Repository Contents

- `streamlit_app.py` - Lightweight dashboard for sample X-ray inspection and demo probability display.
- `Final_Code_Part1.ipynb` - Main model-building notebook.
- `Class Distribution.ipynb` - Class balance exploration.
- `Image Processing/` - Image augmentation notebooks.
- `Data_Splitting/` - Dataset split workflow.
- `Model_Interpretability_LIME.ipynb` - LIME explanation workflow.
- `PARAMETRIC TREND.csv` - Small experiment-tracking CSV.
- `Presentation.pptx` and `Pitching the Project.docx` - Project presentation materials.
- `sample_images/` - Small local reference images used for documentation and quick inspection.

## Dataset

The full chest X-ray dataset is not committed to this repository. Place the dataset locally in a `data/` or `dataset/` directory and update notebook paths as needed before running the training workflow.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Launch the dashboard:

```powershell
streamlit run streamlit_app.py
```

Start Jupyter and open the notebooks:

```powershell
jupyter notebook
```

Recommended order:

1. `Class Distribution.ipynb`
2. `Image Processing/Image Augmentation Part-1.ipynb`
3. `Image Processing/Image Augmentation Part - 2.ipynb`
4. `Data_Splitting/Data Splitting.ipynb`
5. `Final_Code_Part1.ipynb`
6. `Model_Interpretability_LIME.ipynb`

## Notes

- This is an educational ML project, not medical diagnosis software.
- Do not commit full medical datasets, trained model checkpoints, or patient-identifiable data.
- Keep the original group-project attribution visible when reusing or extending this work.
