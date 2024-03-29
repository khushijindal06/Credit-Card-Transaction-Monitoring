# Credit-Card-Transaction-Monitoring
## 1. Introduction:
The Credit Card Fraud Detection project aims to create a powerful machine-learning model capable of detecting fraudulent credit card transactions. Within machine learning, the field of "Anomaly Detection" includes credit card fraud detection. For financial institutions as well as individuals, credit card fraud represents a major financial risk. This project uses the Random Forest algorithm, a potent machine-learning method well-known for its efficiency in classification tasks, to solve this problem. The main objective is to develop a prediction model that minimizes false positives and can detect fraudulent transactions with accuracy.
Credit card fraud poses a significant threat to both financial institutions and cardholders. As technology advances, so do the methods employed by fraudsters. Credit card fraud detection has become a crucial application of data science and machine learning. The goal is to identify and prevent unauthorized or fraudulent transactions in real-time, minimizing financial losses and ensuring the security of financial systems.

## 2) Background:

The increasing prevalence of credit card fraud poses a significant threat to financial institutions and cardholders alike. Traditional methods of fraud detection often struggle to keep pace with the dynamic and sophisticated tactics employed by fraudsters. In this context, we aim to develop an advanced credit card fraud detection system leveraging machine learning algorithms to enhance accuracy and minimize false positives.

## 3) Objectives:

Imbalanced Data Handling:

Address the challenge posed by imbalanced data, where fraudulent transactions represent a minority class, leading to potential bias in the model. Develop strategies to handle this class imbalance and improve model performance.

Model Development:

Utilize machine learning algorithms, such as Random Forest and k-Nearest Neighbors (KNN), to train a robust fraud detection model. Leverage the algorithm's ability to learn patterns from historical data to identify potential fraudulent activities.

Performance Metrics:

Evaluate the effectiveness of the model using appropriate performance metrics, including accuracy, precision, recall, F1-score, and the area under the Receiver Operating Characteristic (ROC) curve. Given the imbalanced nature of the dataset, prioritize metrics that consider false positives and false negatives.

## 4) Dataset overview:
The dataset we have used is downloaded from Kaggle. The dataset comprises transactions that occurred over two days, encompassing 284,807 transactions. Within this dataset are 492 instances of fraud, highlighting the imbalanced nature of fraud occurrences against legitimate transactions.
## Dataset link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## 5) Solution Approach: 
In this project, we explored multiple machine learning algorithms for fraud detection:

Logistic Regression: A baseline algorithm for binary classification tasks. It models the probability of a certain class or event occurring.
Decision Trees: A tree-structured model where internal nodes represent features, branches represent decisions, and leaf nodes represent outcomes. It's a powerful tool for classification tasks.
Random Forest: An ensemble learning method based on decision trees. It builds multiple decision trees and merges their predictions to improve accuracy and reduce overfitting.

## 6) Handling of Imbalanced dataset:
Imbalanced datasets pose a challenge for machine learning models as they tend to bias towards the majority class. In our dataset, fraudulent transactions represent a minority class, making the model prone to overlooking them.

To address this challenge, we employed two techniques: undersampling and SMOTE (Synthetic Minority Over-sampling Technique) analysis.

Undersampling:
Undersampling involves reducing the number of instances in the majority class to balance the class distribution. This technique randomly selects a subset of the majority class samples, typically matching the number of samples in the minority class. By doing so, it helps mitigate the class imbalance issue and prevents the model from being overly biased towards the majority class. However, undersampling may lead to information loss as it discards a significant portion of the majority class data.

SMOTE Analysis:
SMOTE is a widely used oversampling technique that generates synthetic samples of the minority class to balance the dataset. It works by interpolating new instances between existing minority class samples in the feature space. This technique helps in increasing the representation of the minority class without duplication, thus addressing the class imbalance problem. By creating synthetic samples, SMOTE enhances the model's ability to learn from the minority class and improve its predictive performance.

In our approach, we combined both undersampling and SMOTE analysis to ensure a more balanced representation of fraudulent transactions in our training data. This hybrid approach aims to leverage the strengths of both techniques while minimizing their limitations.

## 7) Observations and Results:
After performing oversampling using SMOTE analysis, the RandomForestClassifier achieved remarkable performance metrics:

- Precision Score: 0.9997818776697265
- Accuracy Score: 0.9998909844107707
_- Recall Score: 1.00_
- F1 Score: 0.9998909269392281

After implementing the Random Forest algorithm and performing the hybrid undersampling-SMOTE analysis, we achieved a _100% recall value_, indicating that our model successfully identified all instances of fraudulent transactions without missing any. The recall score, which measures the ability of the model to correctly identify all instances of fraudulent transactions, stands out as particularly noteworthy. With a perfect recall score of 1.0, the model successfully detected every fraudulent transaction in the dataset, demonstrating its exceptional sensitivity to identifying fraudulent activities. This is crucial in fraud detection scenarios where missing even a single fraudulent transaction can have significant financial implications.

