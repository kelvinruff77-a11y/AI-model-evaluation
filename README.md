# 📚 AI Model Evaluation — Text Classification (Product Reviews)

## 📌 Overview
This project trains and evaluates a text classification model (sentiment analysis) on a realistic product reviews dataset.  
The pipeline includes preprocessing, TF-IDF vectorization, Logistic Regression training, evaluation, and visualization (confusion matrix).

## 🧰 Tools & Libraries
- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## ⚙️ Files
- `product_reviews_dataset.csv` — dataset with 100 realistic product reviews (skewed positive)
- `text_classification_model.py` — full training and evaluation script
- `model_predictions.csv` — exported predictions (generated after running the script)
- `confusion_matrix.png` — confusion matrix image (generated after running the script)
- `README.md` — this document

## 🔍 How to run
1. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
2. Run the script from the project folder:
```bash
python text_classification_model.py
```
Outputs will be saved in the same folder: `model_predictions.csv` and `confusion_matrix.png`.

## 🧠 Notes & Takeaways
- Dataset is intentionally realistic (more positive reviews) to mirror real product feedback distributions.
- Logistic Regression provides a fast, interpretable baseline. Consider trying ensemble models or fine-tuning hyperparameters for production.
- Always check class balance and consider stratified sampling (used here) to keep evaluation fair.

## Author
Kelvin Musyoka — Data Analyst & AI Data Annotator
