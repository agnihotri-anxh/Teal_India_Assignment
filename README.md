# Property Address Classification Model

This project solves the task of classifying raw property addresses into predefined categories:
**flat, houseorplot, landparcel, commercial unit, others.**

---

## Project Summary

The dataset consisted of labeled property addresses.  
The goal was to build a machine learning model that generalizes well on unseen addresses.

A classical NLP pipeline was used because the text is short, structured, and keyword-driven rather than semantic.  
TF-IDF with n-grams was found to work best for this type of data.

---

## 🔧 Approach

1. **Preprocessing**
   - Lowercasing
   - Removing special characters
   - Keeping numbers (important for addresses)
   - Normalizing whitespace

2. **Vectorization**
   - TF-IDF  
   - `max_features=2000`  
   - bi-grams (1,2) to capture address patterns like *"flat no"*, *"plot no"*.

3. **Model Training**
   - Evaluated Logistic Regression, Naive Bayes, SVM, and Random Forest.
   - Random Forest performed best on the validation set.

4. **Hyperparameter Tuning**
   - Used `RandomizedSearchCV` to improve accuracy and macro-F1.
   - Final model retrained on train + validation data.

---

## Final Performance (Test Set)

|     Metric     |   Score   |
|----------------|-----------|
| Accuracy       | **0.92**  |
| Macro F1 Score | **0.908** |

Confusion matrix and classification report are included in the notebook.

---

##  Artifacts Included

- `final_rf_model.pkl`
- `tfidf_vectorizer.pkl`
- `label_encoder.pkl` (if used)
- Notebook and script for reproducibility.

---

##  Running Inference

```python
from joblib import load
model = load("final_rf_model.pkl")
vectorizer = load("tfidf_vectorizer.pkl")

def predict(text):
    return model.predict(vectorizer.transform([text]))[0]
