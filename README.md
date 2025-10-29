# Disease_Predictor_Model
# Disease Predictor Model (Diabetes)

A machine learning project that predicts diabetes risk using Random Forest Classifier based on clinical measurements and patient data.

## 📋 Overview

This project builds a predictive model for diabetes diagnosis using the Pima Indians Diabetes Dataset. The model analyzes various health metrics to determine the likelihood of diabetes in patients.

## 🎯 Features

- **Data Analysis**: Comprehensive exploratory data analysis with visualizations
- **Machine Learning**: Random Forest Classifier for disease prediction
- **Feature Importance**: Visualization of which factors contribute most to predictions
- **Model Evaluation**: Accuracy metrics, confusion matrix, and classification report
- **Model Persistence**: Saved model for future predictions

## 📊 Dataset

The project uses the Pima Indians Diabetes Dataset with the following features:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age in years
- **Outcome**: Target variable (0: No diabetes, 1: Diabetes)

**Dataset Size**: 768 samples

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and metrics
- **matplotlib**: Static visualizations
- **seaborn**: Statistical data visualization
- **plotly**: Interactive visualizations
- **joblib**: Model serialization

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/disease-predictor-model.git
cd disease-predictor-model
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn joblib
```

## 🚀 Usage

### Running the Notebook

Open and run the Jupyter notebook:
```bash
jupyter notebook "Disease Predictor Model (1).ipynb"
```

### Using the Trained Model

```python
import joblib
import numpy as np

# Load the saved model
model = joblib.load('disease_prediction_model.pkl')

# Prepare input data (scaled values)
# [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
sample_input = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

# Make prediction
prediction = model.predict(sample_input)

if prediction[0] == 1:
    print("The patient is likely to have Diabetes.")
else:
    print("The patient is unlikely to have Diabetes.")
```

## 📈 Model Performance

- **Accuracy**: ~72.7%
- **Algorithm**: Random Forest Classifier (100 estimators)
- **Train-Test Split**: 80-20

### Classification Report
```
              precision    recall  f1-score   support
           0       0.79      0.79      0.79        99
           1       0.62      0.62      0.62        55
    accuracy                           0.73       154
```

## 🔍 Key Findings

The feature importance analysis reveals:
1. **Glucose** levels are the most important predictor (~26%)
2. **BMI** is the second most significant factor (~17%)
3. **Age** contributes significantly (~14%)
4. **Diabetes Pedigree Function** (family history) is important (~12%)

## 📁 Project Structure

```
disease-predictor-model/
│
├── Disease Predictor Model (1).ipynb    # Main notebook
├── disease_prediction_model.pkl         # Saved model
├── README.md                            # This file
└── requirements.txt                     # Python dependencies
```

## 🔮 Future Improvements

- [ ] Implement hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- [ ] Try additional algorithms (XGBoost, Neural Networks)
- [ ] Add cross-validation for more robust evaluation
- [ ] Handle missing values and zero values more effectively
- [ ] Create a web interface using Streamlit or Flask
- [ ] Add SHAP values for better model interpretability

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

Your Name - Pragya Saha

Project Link: [https://github.com/yourusername/disease-predictor-model]()

## 🙏 Acknowledgments

- Dataset: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Scikit-learn Documentation
- Plotly Graphing Library

---

⭐ If you found this project helpful, please consider giving it a star!
