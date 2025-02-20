
# Online Retail Analysis

This repository contains an analysis of an online retail dataset using Python. The project is implemented in a Jupyter Notebook.

## Overview

The project involves the following steps:
1. Installing required packages.
2. Downloading the dataset.
3. Unzipping the dataset.
4. Loading the dataset into a pandas DataFrame.
5. Exploring and analyzing the dataset.
6. Performing RFM (Recency, Frequency, Monetary) analysis.
7. Conducting customer churn analysis.

## Requirements

- Python 3
- Jupyter Notebook
- pandas
- matplotlib
- openpyxl
- wget
- numpy
- seaborn
- scikit-learn

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/bhushanmandava/Online_Retail_analysis.git
    cd Online_Retail_analysis
    ```

2. Install the required packages:
    ```bash
    pip install pandas matplotlib openpyxl wget numpy seaborn scikit-learn
    ```

## Usage

1. Open the Jupyter Notebook:
    ```bash
    jupyter notebook Online_Retail_Analysis.ipynb
    ```

2. Run the cells in the notebook to perform the analysis.

## Dataset

The dataset used in this project is from the UCI Machine Learning Repository. It contains transactional data of an online retail store.

## Analysis Steps

1. **Package Installation**:
    ```python
    !pip install openpyxl numpy seaborn scikit-learn
    ```

2. **Import Libraries**:
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import KMeans
    ```

3. **Download Dataset**:
    ```python
    !wget https://archive.ics.uci.edu/static/public/352/online+retail.zip
    ```

4. **Unzip Dataset**:
    ```python
    !unzip online+retail.zip
    ```

5. **Load Dataset**:
    ```python
    dataset = pd.read_excel('Online Retail.xlsx', dtype={'InvoiceNo': 'string', 'StockCode': 'string', 'Description': 'string', 'Country': 'string'})
    ```

6. **Explore Dataset**:
    ```python
    dataset.head()
    dataset.shape
    dataset.info()
    dataset.isna().sum()
    dataset[dataset.Description.isna()]
    ```

## RFM Analysis

The project includes an RFM analysis to segment customers based on Recency, Frequency, and Monetary value:
- **Recency**: Days since last purchase
- **Frequency**: Number of purchases
- **Monetary**: Total spend

### Segmenting Customers

```python
# Segment Customers based on RFM
rfm['R_Segment'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
rfm['F_Segment'] = pd.qcut(rfm['Frequency'], 4, labels=[1, 2, 3, 4])
rfm['M_Segment'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])
rfm['RFM_Score'] = rfm[['R_Segment', 'F_Segment', 'M_Segment']].sum(axis=1)
```

### Customer Segmentation

Using RFM scores, customers are segmented into different categories such as:
- **Champions**: High Recency, Frequency, and Monetary value
- **Loyal Customers**: Frequent purchasers
- **At Risk**: High Monetary value but low Recency

## Customer Churn Analysis

The project also includes a churn analysis to identify customers at risk of leaving:
- **Feature Engineering**: Creating features like purchase frequency, average spend, and time since last purchase.
- **Clustering**: Using K-means clustering to identify segments.
- **Visualizations**: Using seaborn for insights into customer behavior.

## Visualizations

The analysis includes visualizations such as:
- RFM distribution plots
- Customer segments using scatter plots
- Heatmaps to understand correlations

## Conclusion

This analysis provides insights into customer behavior, helping businesses to:
- Improve customer retention
- Target marketing campaigns effectively
- Identify loyal customers and potential churn risks
