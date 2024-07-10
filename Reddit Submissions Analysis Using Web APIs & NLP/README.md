# Reddit Submissions Analysis Using Web APIs & NLP

## Executive Summary

This project aimed to classify posts from two subreddits, Soccer and Tennis, using Natural Language Processing (NLP) and machine learning techniques. We collected data using PRAW, performed data preprocessing including text vectorization using TF-IDF, and trained three different models: Logistic Regression, Random Forest, and Support Vector Machine (SVM). Here's a summary of the findings and model performances:

#### Model Performance Comparison

| Model               | Mean Accuracy | Standard Deviation | Accuracy |
|---------------------|---------------|--------------------|----------|
| Logistic Regression | 94.51%        | 1.50%              | 96.1%    |
| Random Forest       | 93.90%        | 1.47%              | 95.6%    |
| SVM                 | 96.57%        | 0.75%              | 97.4%    |

**Key Insights**:
- **Logistic Regression**: Achieved an accuracy of 96.1% with a mean accuracy of 94.51% and showed robust performance.
- **Random Forest**: Demonstrated an accuracy of 95.6% with a mean accuracy of 93.90%. It exhibited consistent performance but slightly lower accuracy compared to Logistic Regression.
- **SVM**: Outperformed other models with the highest accuracy of 97.4% and the lowest standard deviation of 0.75%. SVM's ability to handle high-dimensional data and text classification tasks contributed to its superior performance.

## Project Overview: 
The project involves using NLP and machine learning to classify Reddit posts. By gathering data from two subreddits, we aim to train a model that can accurately determine which subreddit a given post comes from.

Goal: Classify posts from two subreddits using Natural Language Processing (NLP) and machine learning.
Subreddits chosen: 
- Soccer
- Tennis

## DATA COLLECTION

Data Source: Reddit
Method: PRAW API (Python Reddit API Wrapper)
Collected posts from Soccer and Tennis subreddits.
Number of posts: 3000

### DataFrame Column Descriptions

Here is a table showing the columns, their data types, and descriptions:

| Column Name   | Data Type | Description                                      |
|---------------|-----------|--------------------------------------------------|
| created_utc   | int64     | The UTC timestamp when the post was created      |
| title         | object    | The title of the Reddit post                     |
| self_text     | object    | The body text of the Reddit post (if available)  |
| subreddit     | object    | The subreddit from which the post was collected  |

## Data Cleaning & Preprocessing
- Feature Engineering

- Duplicate Rows

- Tokenization and removal of stop words

- Vectorized Text Data Using Inverse Document Frequency (TF-IDF) (TfidfVectorizer)


## Exploratory Data Analysis
- Most Common Words In Each Subreddit
- Topic Distribution by Subreddit.
- Sentiment Distribution by Subreddit

## Model Selection

- Logistic Regression
- Random Forest
- SVM
- Used GridSearchCV for hyperparameter tuning
- Evaluated using cross-validation


### Logistic Regression

**Cross-validation Mean Accuracy:** 94.51%
**Standard Deviation:** 1.50%
**Brief Evaluation of Performance:**
Logistic Regression performed well, with a mean accuracy of 94.51%. It is a simple and interpretable model, but it has a high standard deviation.


**Confusion Matrix**:

|            | Predicted: Soccer (0) | Predicted: Tennis (1) |
|------------|------------------------|-----------------------|
| Actual: Soccer (0) | 382                    | 3                     |
| Actual: Tennis (1) | 22                     | 249                   |

**Precision, Recall, and F1-score**:

| Class        | Precision | Recall | F1-score |
|--------------|-----------|--------|----------|
| Soccer (0)   | 0.95      | 0.99   | 0.97     |
| Tennis (1)   | 0.99      | 0.92   | 0.95     |

**Accuracy**: 96.1%


### Random Forests Model

**Cross-validation Mean Accuracy**: 93.90%
**Standard Deviation**: 1.47%

**Brief Evaluation of Performance**:
Random Forest showed strong performance with a mean accuracy of 93.90%. It is robust and handles overfitting well due to its ensemble nature. The standard deviation of 1.47% indicates consistent performance across folds.

**Confusion Matrix**:

|            | Predicted: Soccer (0) | Predicted: Tennis (1) |
|------------|------------------------|-----------------------|
| Actual: Soccer (0) | 359                    | 26                     |
| Actual: Tennis (1) | 3                     | 268                   |

**Precision, Recall, and F1-score**:

| Class        | Precision | Recall | F1-score |
|--------------|-----------|--------|----------|
| Soccer (0)   | 0.99      | 0.93   | 0.96     |
| Tennis (1)   | 0.91      | 0.99   | 0.95     |

**Accuracy**: 95.6%

### Support Vector Machine (SVM)

**Cross-validation Mean Accuracy**: 96.57%
**Standard Deviation**: 0.75%

**Brief Evaluation of Performance**:
SVM outperformed the other models with a mean accuracy of 96.57%. It demonstrated the most consistent performance, as shown by the lowest standard deviation of 0.75%. SVM effectively handles high-dimensional data and performs exceptionally well with text classification tasks.

**Confusion Matrix**:

|            | Predicted: Soccer (0) | Predicted: Tennis (1) |
|------------|------------------------|-----------------------|
| Actual: Soccer (0) | 381                    | 4                     |
| Actual: Tennis (1) | 13                     | 258                   |

**Precision, Recall, and F1-score**:

| Class        | Precision | Recall | F1-score |
|--------------|-----------|--------|----------|
| Soccer (0)   | 0.97      | 0.99   | 0.98     |
| Tennis (1)   | 0.98      | 0.95   | 0.97     |

**Accuracy**: 97.4%


### MODEL COMPARISON

| Model               | Mean Accuracy | Standard Deviation | Accuracy |
|---------------------|---------------|--------------------|----------|
| Logistic Regression | 94.51%        | 1.50%              | 96.1%    |
| Random Forest       | 93.90%        | 1.47%              | 95.6%    |
| SVM                 | 96.57%        | 0.75%              | 97.4%    |

**Summary**:
SVM achieved the highest accuracy of 97.4% and demonstrated the most consistent performance with a standard deviation of 0.75%. Logistic Regression and Random Forest also performed well but were slightly less accurate and consistent compared to SVM. SVM's effectiveness in handling high-dimensional data and text classification tasks contributed to its superior performance in this analysis.

### Conclusion

SVM outperforms other models for this Dataset.
Potential applications for classifying Reddit posts.
Future work: Expanding to more subreddits, exploring additional features.
