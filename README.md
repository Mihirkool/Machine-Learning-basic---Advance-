# Fundamentals of Machine Learning – Experiments Repository

This repository contains a complete collection of experiments covering the **Fundamentals of Machine Learning**, progressing from basic regression to advanced neural and kernel-based methods.  
Each experiment is implemented using Python and demonstrated on the **Weather Prediction Dataset**, making the concepts practical and relatable.

---

## 📘 Overview

The main goal of this repository is to provide a structured, hands-on learning experience for understanding how various machine learning algorithms work — from linear regression to multilayer perceptrons and RBF networks.  
Every experiment includes step-by-step code, clear mathematical intuition, and result visualization using real-world **weather prediction data**.

---

## 🧠 Experiments Included

### Regression Models
- **Simple Linear Regression** – Predicting temperature or humidity using a single weather parameter.  
- **Multivariate Linear Regression** – Forecasting weather metrics using multiple atmospheric features.


### Clustering
- **K-Means Clustering** – Grouping weather patterns based on temperature, humidity, and pressure similarity.

### Support Vector Machines (SVM)
- Implemented for both **linear** and **non-linear (kernel)** classification tasks.

### Probabilistic Learning
- **Naïve Bayes Classifier** – Probabilistic approach for predicting categorical weather conditions.

### Decision Trees
- **Decision Tree Classifier** – Interpretable model for weather condition prediction.  
- **Pruning Techniques** – Reducing model complexity and preventing overfitting.

### Neural Networks
- **Multilayer Perceptron (MLP)** – Feedforward neural network for weather trend prediction.  
- **Radial Basis Function Neural Network (RBFNN)** – Kernel-based network for non-linear weather relationships.

### Kernel Methods
- **Kernel Functions** – Exploring polynomial and Gaussian (RBF) kernels for improved non-linear learning.

---

## 🌦️ Dataset Used: Weather Prediction Data

The dataset contains various meteorological attributes such as:  
- Temperature  
- Humidity  
- Wind Speed  
- Pressure  
- Rainfall Indicators  

It is used across all experiments for consistency and comparative understanding of different algorithms.

---

## 🛠️ Tools & Libraries Used

- Python 3.x  
- NumPy  
- Pandas  
- Matplotlib / Seaborn  
- Scikit-learn  
- TensorFlow / Keras  

---

## 📂 Repository Structure

📁 fundamentals-of-ml/
├── Dataset/
│ └── Weather_Prediction_Data.csv
├── Linear_Regression/
│ ├── Simple_Linear_Regression.ipynb
│ └── Multivariate_Linear_Regression.ipynb
├── Logistic_Regression/
│ ├── Simple_Logistic_Regression.ipynb
│ └── Multivariate_Logistic_Regression.ipynb
├── KMeans/
│ └── KMeans_Clustering.ipynb
├── SVM/
│ └── Support_Vector_Machine.ipynb
├── Bayesian/
│ └── Naive_Bayes_Classifier.ipynb
├── Decision_Tree/
│ ├── Decision_Tree.ipynb
│ └── Pruning.ipynb
├── Neural_Networks/
│ ├── MLP.ipynb
│ └── RBFNN.ipynb
├── Kernel_Methods/
│ └── Kernel_Functions.ipynb
└── README.md


---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/fundamentals-of-ml.git


Navigate to the directory:

cd fundamentals-of-ml


Install dependencies:

pip install -r requirements.txt


Open Jupyter Notebook:

jupyter notebook


Run any experiment notebook to see results using the weather dataset.

🎯 Learning Outcomes

Understand and implement various ML algorithms.

Learn model training, evaluation, and tuning using real-world data.

Gain clarity on both supervised and unsupervised learning concepts.

Develop a foundation for advanced models like SVMs and Neural Networks.

🧾 License

This project is open-source and available under the MIT License
.

👨‍💻 Author

Mihir
Machine Learning Enthusiast | Developer | Research Learner

---


