# 🚀 Advanced Customer Segmentation Tool

ONLine: https://jayant-chand-ibm-project-market-segmentation-dec-app-1vjdcn.streamlit.app/

## 📌 Overview
This Streamlit-based web application allows users to perform **customer segmentation** using an advanced clustering approach that integrates **Deep Embedded Clustering (DEC) with K-Means. The app provides an interactive interface for data preprocessing, PCA-based dimensionality reduction, autoencoder training, and cluster visualization.

✨ Features
- 📂 Upload CSV dataset for segmentation
- 📊 Data preprocessing including handling missing values and encoding categorical features
- 🔥 PCA for dimensionality reduction (retains 95% variance)
- 🤖 Autoencoder for feature extraction
- 🔍 K-Means clustering with a user-defined number of clusters
- 📈 Visualizations
  - 2D and 3D cluster representation
  - Customer group distribution
  - Heatmap of cluster characteristics
  - Scatter matrix for feature relationships
  - Radar chart for group strengths

## 🏗️ Installation
### 1️⃣ Clone the Repository**
```bash
git clone https://github.com/jayant-chand/IBM_project_Market_Segmentation_DEC.git
cd customer-segmentation-app
```
### 2️⃣ Create Virtual Environment**
```bash
python -m venv env
source env/bin/activate  # For Mac/Linux
env\Scripts\activate  # For Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage
Run the application using:
```bash
streamlit run app.py
```
Upload a CSV file set the number of clusters, and explore interactive visualizations!

## 🛠️ Technologies Used
- Python
- Streamlit (UI Framework)
- TensorFlow & Keras (Autoencoder Model)
- Scikit-learn (PCA & K-Means)
- Plotly (Visualizations)

## 📜 License
This project is licensed under the MIT License.

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss your proposed changes.

---
🔗 **Developed by: Jayant Chand and Team | ✉️ Contact: jayantchand2379@gmail.com

