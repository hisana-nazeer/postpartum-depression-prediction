# Postpartum Depression Prediction Model

A clinical ML pipeline that predicts postpartum depression (PPD) risk from patient symptom data, deployed as a REST API with a web interface.

**Live Demo:** [ppd-assessment.netlify.app](https://ppd-assessment.netlify.app)  
**API Endpoint:** [postpartum-depression-prediction-2.onrender.com](https://postpartum-depression-prediction-2.onrender.com)

---

## Project Overview

Postpartum depression affects 1 in 5 new mothers yet remains widely underdiagnosed. This model provides an accessible, explainable screening tool that can flag at-risk individuals before they reach crisis point — built as the clinical risk-flagging backbone of [Nurtural](https://nurtural.app), a postpartum wellness app.

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Data processing | Python, Pandas |
| ML model | scikit-learn, Random Forest |
| Class imbalance | SMOTE (imbalanced-learn) |
| Explainability | SHAP (TreeExplainer) |
| API | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Render (API), Netlify (frontend) |

---

## Dataset

- 1,503 patient survey responses
- 10 clinical features based on Edinburgh Postnatal Depression Scale (EPDS)
- No independent target variable — engineered using EPDS-inspired clinical scoring logic

---

## Methodology

### Target Variable Engineering
PPD label created from 3 core mood symptoms (sadness, anxiety, guilt) using a proportionally scaled EPDS threshold. Features and target kept separate to prevent data leakage.

### Feature Engineering
- Ordinal encoding based on clinical severity (not arbitrary numeric assignment)
- `Not interested to say` on suicide attempt treated as a positive signal
- Age ranges converted to midpoints for continuous representation
- Mode imputation for missing values (mean is meaningless for discrete ordinal data)

### Handling Class Imbalance
SMOTE applied exclusively to the training set — never the test set — to prevent synthetic data from contaminating evaluation.

### Model Selection
Two models trained and compared:

| Model | Recall | F1 | Accuracy |
|-------|--------|-----|---------|
| Logistic Regression | 0.62 | 0.72 | 0.63 |
| Random Forest | **0.91** | **0.84** | **0.93** |

**Primary metric: Recall** — in a clinical screening tool, a false negative (missing a real PPD case) carries far greater cost than a false positive.

---

## SHAP Explainability

Used SHAP TreeExplainer to identify which features drive predictions most:

1. Irritable towards baby & partner — most impactful
2. Problems bonding with baby
3. Trouble sleeping at night
4. Age — least impactful

Key insight: high irritability actually reduces PPD risk in this model because PPD was defined by core mood symptoms. This reveals how target variable design directly shapes what a model learns.



---

## API Usage

**Base URL:** `https://postpartum-depression-prediction-2.onrender.com`

### Health Check
```
GET /
```

### Predict
```
POST /predict
Content-Type: application/json

{
  "Age": 37.5,
  "Irritable_towards_baby_and_partner": 2,
  "Trouble_sleeping_at_night": 2,
  "Problems_concentrating_or_making_decision": 1,
  "Overeating_or_loss_of_appetite": 1,
  "Problems_of_bonding_with_baby": 2,
  "Suicide_attempt": 0
}
```

### Response
```json
{
  "prediction": 1,
  "probability": 0.847,
  "result": "PPD Positive"
}
```

---

## Project Structure

```
├── PPD_final.ipynb       # Full ML pipeline
├── app.py                # FastAPI server
├── ppd_clean.csv         # Preprocessed dataset
├── ppd_model.pkl         # Trained Random Forest model
├── ppd_scaler.pkl        # Fitted StandardScaler
├── ppd_frontend.html     # Web interface
├── requirements.txt      # Dependencies
├── Procfile              # Render deployment config
└── shap_summary.png      # SHAP feature importance plot
```

---

## Key Engineering Decisions

**Why Recall over Accuracy?**  
For a healthcare screening tool, missing a real PPD case (false negative) is clinically more dangerous than a false alarm (false positive). Recall directly measures this.

**Why SMOTE on training data only?**  
Applying SMOTE before splitting would leak synthetic data into the test set, making evaluation dishonest. SMOTE is applied after splitting, on training data only.

**Why not use the full dataset for target creation?**  
Using all symptom columns to both create the target variable and train the model creates a mathematical relationship that trivially predicts at 100% accuracy — data leakage. Target was engineered from 3 core mood symptoms; remaining features used for training.

---

## Built With

This model serves as the risk-flagging backend for **Nurtral** — a postpartum wellness app providing AI-powered support for new mothers (currently in progress).

---

## Author

**Hisana Nazeer**  
[LinkedIn](https://linkedin.com/in/hisana-nazeer) | [GitHub](https://github.com/hisana-nazeer)
