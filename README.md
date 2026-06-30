# Amazon Product Price Prediction

Predicting Amazon product prices using **text, images, and machine learning**.

This project was built for the **Amazon ML Challenge 2025**. The goal is to estimate the price of a product from its title, description, brand, unit, and product image.

---

## Overview

Instead of relying on only product text, this project combines:

* Product title
* Bullet points
* Product description
* Brand name
* Product unit
* Product image

The final model learns from all these features together to make more accurate price predictions.

---

## Dataset

* **75,000+** training products
* Product metadata
* Product images
* Target: Product price

---

## Features Used

### Text Features

* Product title
* Bullet points
* Product description
* TF-IDF Vectorization

### Categorical Features

* Brand name
* Product unit

### Image Features

* EfficientNet_b0 (PyTorch)
* Image embeddings extracted from product images

---

## Models

* LightGBM
* CatBoost
* XGBoost
* Weighted Ensemble (LightGBM + CatBoost)

---

## Results

### LightGBM

| Metric |     Score |
| ------ | --------: |
| MAE    | **11.42** |
| RMSE   | **33.04** |
| R²     | **0.277** |

### CatBoost

| Metric |     Score |
| ------ | --------: |
| MAE    | **12.08** |
| RMSE   | **33.63** |
| R²     | **0.251** |

### Best Validation Score

**SMAPE:** **50.14%**

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* PyTorch
* LightGBM
* CatBoost
* XGBoost
* OpenCV
* Matplotlib

---

## Project Structure

```
├── data/
├── models/
├── notebooks/
└── README.md
```

---

## Key Learnings

* Built an end-to-end machine learning pipeline from data preprocessing to model evaluation.
* Learned how to combine text, image, and categorical features for a multimodal regression task.
* Compared multiple gradient boosting models and evaluated their strengths.
* Used transfer learning (EfficientNet_b0) to extract meaningful image representations.
* Improved prediction quality through feature engineering and ensemble learning.

---

## Future Improvements

* Fine-tune vision-language models such as CLIP or SigLIP.
* Use transformer-based text embeddings instead of TF-IDF.
* Perform hyperparameter optimization with Optuna.
* Experiment with advanced stacking and blending techniques.

---

## How to Run

```bash
git clone <repository-url>

cd amazon_ml_challenge_25

pip install -r requirements.txt

jupyter notebook
```

Run the notebook to:

1. Preprocess the data
2. Extract image embeddings
3. Train regression models
4. Generate predictions
5. Create the submission file

---

## License

This project is for learning and research purposes. The dataset belongs to the Amazon ML Challenge organizers.

---

**Simple project summary**

> Built an end-to-end multimodal machine learning pipeline that predicts Amazon product prices using product text, categorical attributes, and image embeddings. Compared LightGBM, CatBoost, and XGBoost models, and improved performance through feature engineering and ensemble learning.
