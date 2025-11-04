# Customer Behavior & Shopping Analysis

A machine learning project that predicts the most sold product category based on consumer behavior and shopping patterns.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objective](#objective)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project analyzes consumer behavior and shopping habits to predict product category sales. By leveraging machine learning algorithms, we can help businesses understand customer preferences and optimize their sales strategies based on demographics, purchase behavior, and promotional effectiveness.

## 📊 Problem Statement

The consumer behavior and shopping dataset contains various features such as customer demographics, purchase behavior, and product details. It provides comprehensive insights into consumers' preferences, tendencies, and patterns during their shopping experiences. 

This dataset aims to provide an understanding of consumer behavior in purchasing products according to their ages and gender, including the impact of promotional ads and subscription plans. Developing an effective prediction model can enhance sales strategies in the market.

## 🎯 Objective

To develop a machine learning model that predicts the most sold product category based on customer demographics, shopping patterns, and promotional factors.

## 📁 Dataset

**Source:** [Kaggle - Consumer Behavior and Shopping Habits Dataset](https://www.kaggle.com/datasets/zeesolver/consumer-behavior-and-shopping-habits-dataset/data)

**Dataset Statistics:**
- **Total Records:** 3,900 rows
- **Total Features:** 18 columns
- **Data Types:** Mixed (Numerical and Categorical)

### Features Description

| Feature | Type | Description |
|---------|------|-------------|
| Customer ID | Numerical | Unique identifier for each customer |
| Age | Numerical | Customer's age |
| Gender | Categorical | Customer's gender |
| Item Purchased | Categorical | Specific item bought |
| **Category** | Categorical | **Product category (Target Variable)** |
| Purchase Amount (USD) | Numerical | Transaction amount in USD |
| Location | Categorical | Customer's location |
| Size | Categorical | Product size |
| Color | Categorical | Product color |
| Season | Categorical | Season of purchase |
| Review Rating | Numerical | Customer rating (2.5-5.0) |
| Subscription Status | Categorical | Active subscription (Yes/No) |
| Shipping Type | Categorical | Delivery method |
| Discount Applied | Categorical | Discount status (Yes/No) |
| Promo Code Used | Categorical | Promo code usage (Yes/No) |
| Previous Purchases | Numerical | Number of past purchases |
| Payment Method | Categorical | Payment type used |
| Frequency of Purchases | Categorical | Purchase frequency pattern |

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository
```bash
git clone https://github.com/Devikrishna545/CustomerBehaviour_ShoppingAnalysis.git
cd CustomerBehaviour_ShoppingAnalysis
```

2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

### Required Libraries
```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

## 🚀 Usage

1. Launch Jupyter Notebook
```bash
jupyter notebook
```

2. Open the main notebook
```
E-Commerce Dataset ML_New.ipynb
```

3. Run all cells sequentially to:
   - Load and explore the dataset
   - Perform data preprocessing
   - Conduct feature selection
   - Train the model
   - Evaluate model performance

## 📈 Model Performance

### Algorithms Used

1. **Feature Selection:**
   - Random Forest Classifier
   - Lasso Regression

2. **Classification Model:**
   - Random Forest Classifier

### Results

- **Accuracy:** 84.5%
- **Model:** Random Forest Classifier

### Key Insights

- The model successfully predicts product categories with high accuracy
- Random Forest proved effective for handling mixed data types
- Feature selection improved model performance and interpretability

## 💻 Technologies Used

- **Programming Language:** Python 3.8+
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Development Environment:** Jupyter Notebook

## 📂 Project Structure

```
CustomerBehaviour_ShoppingAnalysis/
│
├── E-Commerce Dataset ML_New.ipynb  # Main analysis notebook
├── shopping_behavior.csv             # Dataset (not included in repo)
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore file
│
└── results/                          # (Optional) Model outputs and visualizations
    ├── models/                       # Saved models
    └── figures/                      # Generated plots
```

## 🔍 Results

The project successfully:
- ✅ Analyzed 3,900 customer records across 18 features
- ✅ Identified key factors influencing category purchases
- ✅ Achieved 84.5% prediction accuracy
- ✅ Provided actionable insights for sales strategy optimization

### Key Findings

- Category distribution shows Clothing as the most purchased category (1,737 items)
- Customer demographics significantly influence purchase patterns
- Promotional factors (discounts, promo codes) impact buying behavior
- Subscription status correlates with purchase frequency

## 🔮 Future Improvements

- [ ] Implement additional classification algorithms (XGBoost, LightGBM)
- [ ] Perform hyperparameter tuning for better accuracy
- [ ] Add cross-validation for robust model evaluation
- [ ] Create interactive visualizations with Plotly/Dash
- [ ] Deploy model as a web application
- [ ] Incorporate time-series analysis for seasonal trends
- [ ] Add customer segmentation analysis
- [ ] Implement recommendation system

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Devikrishna545**

- GitHub: [@Devikrishna545](https://github.com/Devikrishna545)
- Project Link: [https://github.com/Devikrishna545/CustomerBehaviour_ShoppingAnalysis](https://github.com/Devikrishna545/CustomerBehaviour_ShoppingAnalysis)

## 🙏 Acknowledgments

- Dataset provided by [Kaggle - zeesolver](https://www.kaggle.com/datasets/zeesolver/consumer-behavior-and-shopping-habits-dataset/data)
- Inspiration from e-commerce analytics and consumer behavior research
- Open-source community for excellent ML libraries

---

**Note:** Download the dataset from the Kaggle link provided above and place it in the project root directory before running the notebook.

⭐ If you find this project helpful, please consider giving it a star!
