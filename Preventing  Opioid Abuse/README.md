# Project 5: Group Project - Opioid Abuse
---
Corey Jesmer, Ayodeji Ejigbo & Sarah Roe

## Problem Statement
---

Can we build a model that predicts the likelihood of a patient abusing prescribed pain medication?

## Questions
---

1. Can we predict the likelihood of a patient abusing prescription pain medication based on past addiction history?
2. How do different characteristics correlate to our predictions?
3. How accurate of a model can we build and how could it be improved upon?
4. What type of model gets us the best result?

## Data Dictionary
---


| Feature             | Type   | Dataset | Description                                                                       | Note |
|---------------------|--------|---------|-----------------------------------------------------------------------------------|------|
| **PRLMISAB**        | *int*  | All     | Abused prescription medications (Binary).                       | **TARGET VARIABLE** |    
| **YEAR**            | *int*  | All     | Year of survey (Integer).                                                         |   15=2015, 16=2016, 17=2017   |
| **AGECAT**          | *int*  | All     | Age category (Categorical).                                                       | 1=12-17 years, 2=18-25, 3=26-35, 4=36-49, 5=50 and older |
| **SEX**             | *int*  | All     | Gender (Categorical).                                                             |   0=Male, 1=Female   |
| **MARRIED**         | *int*  | All     | Marital status (Categorical).                                                     |  0=unmarried, 1=divorced, 2=widowed, 3=married     |
| **EDUCAT**          | *int*  | All     | Education level (Categorical).                                                    |   1=h.s. or Less, 2=h.s. grad., 3=some college, 4=college grad.    |
| **EMPLOY18**        | *int*  | All     | Employment status (Categorical).                                                  |  1=not employed, 2=part-time, 3=full-time    |
| **CTYMETRO**        | *int*  | All     | City or metropolitan area (Categorical).                                          |    1=rural, 2=small, 3=large |
| **HEALTH**          | *int*  | All     | Self-reported health (Categorical).                                               |   Likert scale: 1-10   |
| **MENTHLTH**        | *int*  | All     | Self-reported mental health (Categorical).                                        |   Likert scale: 1-10   |
| **PRLMISEVR**       | *int*  | All     | Ever misused prescription medication (Binary).                                    |       |
| **PRLANY**          | *int*  | All     | Misused or abused minimum prescription medications (Binary).                      |       |
| **HEROINEVR**       | *int*  | All     | Ever used heroin (Binary).                                                        |       |
| **HEROINUSE**       | *int*  | All     | Used heroin in past year (Binary).                                                |   Likert scale: 0-5   |
| **TRQLZRS**         | *int*  | All     | Used tranquilizers in past year (Binary).                                         |   Likert scale: 0-5    |
| **SEDATVS**         | *int*  | All     | Used sedatives in past year (Binary).                                             |   Likert scale: 0-5    |
| **COCAINE**         | *int*  | All     | Used cocaine in past year (Binary).                                               |   Likert scale: 0-5    |
| **AMPHETMN**        | *int*  | All     | Used amphetamines in past year (Binary).                                          |  Likert scale: 0-5   |
| **HALUCNG**         | *int*  | All     | Used hallucinogens in past year (Binary).                                         |  Likert scale: 0-5    |
| **TRTMENT**         | *int*  | All     | Received treatment for substance use (Binary).                                    |    Likert scale: 1-10    |
| **MHTRTMT**         | *int*  | All     | Received treatment for mental health (Binary).                                    |   Likert scale: 0-5   |
| **AGE_MARRIED**         | *int*  | All     | Interaction (*) of AGECAT and MARRIED.                                               |       |
| **AGE_EMPLOY**        | *int*  | All     | Interaction (*) of AGECAT and EMPLOY18.                                          |     |
| **REPORTED_HEALTH**         | *int*  | All     | Interaction (*) of HEALTH and MENTHLTH.                                         |      |
| **HISTORY**         | *int*  | All     | Interaction (*) of HEROINEVR and PRLMISEVR                                    |        |
| **UPPERS**         | *int*  | All     | Use of uppers in last year (Cocaine and Amphetamines).                                    |      |
| **DOWNERS**         | *int*  | All     | Use of downers in the last year (Sedatives, Tranqs, Heroin).    

## Data Used
---
1. prlmis-data-full.csv
2. pain_clean.csv
3. Shiverick-PRLMIS-final.pdf
4. Sarah_cleaned_csv


Links to outside resources used:

1. Kaggle Dataset on Opioid Abuse: https://www.kaggle.com/datasets/thedevastator/predicting-pain-reliever-misuse-abuse/data
2. Dataset Documentation that Provided Data Dictionary Context: https://zenodo.org/records/2301844#.Y8OqptJBwUE
3. Opioid Crisis in Young Americans https://murphy.house.gov/media/press-releases/murphy-fentanyl-killing-more-young-americans-covid-19
4. Drug Overdose Death Rates: https://nida.nih.gov/research-topics/trends-statistics/overdose-death-rates#:~:text=Nearly%20108%2C000%20persons%20in%20the,drugs%20from%201999%20to%202022




### Requirements
---
- Python, Jupyter
- Pandas, Numpy, Matplotlib, Seaborn
- GradientBoost, XGB, RandomOverSampler
- Scikit Learn Libraries:
   - StandardScaler, Train Test Split, Metrics, LogisticRegression
   - Pipeline, RandomForestClassifier, GradientBoostingClassifier
   - classification_report, accuracy_score, confusion_matrix
   - PCA, GridSearchCV
- Tensorflow, Keras
   - Conv2D, MaxPooling2D, Flatten, Dense, Dropout
   - Sequential, EarlyStopping

### Executive Summary
---
 This project aims to develop a machine learning model to predict the likelihood of a patient abusing pain medications. This information would allow doctors to make decisions on what types of pain meds to give, how much to give at a time, prevent cases of addiction and/or relapse and
intervene early, create treatment plans and overall improve quality of care for all patients.

 
### Objectives
---
1. Data Processing: Clean and preprocess the text data to ensure it is suitable for model training.
2. Data Exploration: Explore data for correlations to our target variable and create interaction terms.
3. Model Development: Develop and train machine learning models to predict the likelihood of a patient abusing prescription pain medications.
4. Evaluation: Assess the performance of the models using various metrics to ensure their accuracy and reliability.


### Methods
---

#### **Data Cleaning**
The data seemed relatively clean at first glance, however when doing some EDA, a few of the columns had incorrect values. MARRIED, EDUCAT, and CTYMETRO columns had values that were not listed in the data dictionary. We were unable to determine what the values meant, so they were dropped while modeling. The EMPLOY18 column was listed as 0, 1, 2 rather than 1, 2, 3 as was indicated in the data dictionary. As a team we chose to move the 0 column to the 3 column because that would make the largest group ‘Employed’ individuals. Our target variable was the PRLMISEVR which was binary of 0 (no drug abuse) and 1 (drug abuse). There was a large class imbalance with 90% no drug abuse (0) to 10% drug abuse (1), so bootstrapping was implemented to bring the data to a 50(0)-50(1) class imbalance. The columns ‘PRLANY’ and ‘PRLMISAB’ are heavily correlated to our target value so they were also dropped during modeling.

#### **Feature Engineering**
We created new features that were interaction terms of existing features, such as UPPERS (interaction of stimulant category drug usage in last year), DOWNERS (interaction of sedative category drug usage in last year), REPORTED_HEALTH (interaction of self reported physical and mental health of patient), etc.

#### **Instantiating PCA + Multiple Classifier Models**
A principal component analysis (PCA) model was instantiated to determine how many components were necessary and the most important principles in this data set. Next, a PCA with the previously determined components was put into a pipeline along with a random forest classifier, logistic regression, and naive bayes classifier. This pipeline contained 4 pipelines – PCA (n_components=15) + random forest (n_estimators=600, max_features=9, max_depth=30), PCA (same as before) + logistic regression (max_iter=10000, penalty=None), logistic regression (same as before), and naive bayes. The goal was to run the four models and determine the best training and testing score out of the four.

#### **Instantiating Neural Net Model**
A neural net was instantiated with a total of 6 dense hidden layers, two dropout layers and a regularization of 0.001 l2. The layers were as follows:
Dense(32 neurons, Relu activation, l2 regularization)
Dense(32 neurons, Relu activation)
Dropout(0.2)
Dense(64, Relu activation)
Dense(64, Relu activation)
Dropout(0.25)
Dense(128, Relu activation)
Dense(128, Relu activation)
Then a dense output layer with sigmoid activation and 1 neuron (binary classification). In total, this provided 32,801 trainable parameters. When fitting the neural net, a batch size of 32 was used with 100 epochs.

#### **Baseline**
After bootstrapping/oversampling, there was a baseline accuracy of 0.499 drug abuse to no drug abuse.

#### **PCA + Multiple Classifier Models**
Of the four models run, the PCA + random forest model was most successful with a training accuracy of 85.92% and a testing accuracy of 83.35%. Next, the PCA + logistic regression and plain logistic regression performed almost exactly the same on the training data with an accuracy of 73.99%. However, the PCA + logistic regression model performed 0.01% higher on testing accuracy with an accuracy of 74.20%. Lastly, the naive bayes model performed the worst with a training accuracy of 73.75% and testing accuracy of 73.80%.

#### **Neural Net**
The neural net had a slightly higher score compared to the logistic regression models. This model had a training accuracy of 75.03% and a testing accuracy of 75.35%. It also had a training loss of 0.538 and testing loss of 0.536.
  

#### Findings
  - Surprisingly, the correlation between opioid abuse and drugs in the stimulant(UPPERS) category was stronger than the correlation between opioid abuse and drugs in the sedative(DOWNERS) category.

### **Discussion/Conclusion**
---
The increased misuse and abuse of pain medication in recent years has underscored a pressing need for improved pain management treatment. While it’s crucial to prioritize the comfort and recovery of patients, there is a fine balance between delivering relief and minimizing the potential for misuse. These machine learning models aid in reducing judgment calls and provide physicians with insights to make informed decisions about pain management, preventing addiction, reducing the risk of relapse, and elevating overall quality of care.
Classification models can indeed be valuable for predicting drug abuse risk. However, there is a trade-off between a model's complexity, its performance, and its interpretability. These factors are crucial when implementing a machine learning model in practice, as both doctors and patients value understanding how and why decisions are made. While random forests are more complex and less interpretable than logistic regression, they are less opaque than neural networks. In general, with this dataset in particular, random forests combined with PCA demonstrated better accuracy compared to neural networks and logistic regression.
While each model outperformed the baseline predictions, both Type I and Type II errors were present, which could have serious clinical consequences. False negatives, in particular, pose a significant risk by potentially overlooking patients who are at risk for drug abuse or are currently in remission, thereby failing to provide timely intervention. Similarly, false positives may lead to unnecessary concern or intervention for individuals who are not at risk. To address these issues and improve clinical outcomes, it is crucial to continuously refine and validate the models with new and diverse datasets. This ongoing process will help to capture emerging patterns in addiction risk, enhance the models' predictive accuracy, and ultimately provide more reliable tools for identifying and managing drug abuse risk.
Implementation of these models could prove significant in addressing the current opioid epidemic. As of 2023, approximately 75% of the 108,000 drug overdoses involved opioids. While this isn’t a solution, combining predictive modeling with clinical judgment can lead to improved patient care, enhanced patient safety, and reduce the risk of drug abuse/misuse.

### Next steps
---

#### Enhance Data Collection:

1. Gather detailed information on patients' self-reported pain levels to improve the identification of the need for pain medications.
Aim to collect more comprehensive data, reducing the occurrence of missing values across various features.
Data Validation:

2. Contact the dataset creator to clarify discrepancies between the data dictionary and the actual values in the dataset.
Ensure a clear understanding of all features and their respective values.
Model Improvement:

3. Focus on minimizing false negatives in future models to ensure that patients who are at high risk are accurately identified, reducing the likelihood of them being categorized as low risk.