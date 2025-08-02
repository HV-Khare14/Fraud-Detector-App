# Fraud-Detector-App
🔍 Job Fraud Detector | Anveshan Hackathon Project An ML-powered Streamlit web app that detects fake job listings using natural language processing, feature engineering, and XGBoost. Includes real-time predictions, SHAP explanations, search, filtering, and CSV export — built to protect job seekers from online scams

training data csv:- https://drive.google.com/file/d/1oxygynnHOAU3NU3S8xHRdBS9WA_K1E7y/view?usp=sharing

Fraud Detector: Smart Job Scam Detection System

🧠 Project Overview Online job platforms are frequently targeted by scammers. These fake job listings waste applicants’ time and put their personal information at risk.

This project builds a machine learning pipeline that automatically detects fraudulent job listings and presents insights in a Streamlit-powered interactive dashboard.

🎯 Problem Statement Develop a system to classify job listings as fraudulent or genuine using structured and unstructured data. Display predictions and insights in an intuitive dashboard.

🚀 Automatic Job Fraud Detection Classifies job listings as real or fake using advanced machine learning techniques.

🧠 Hybrid Feature Engineering Combines TF-IDF text vectors with structured numeric features:

• Description length & word count

• Suspicious keywords (e.g., money, bitcoin)

• Free email domain indicators (e.g., gmail.com, yahoo.com)

• Digit count in job titles

🌲 XGBoost Classifier for Accuracy & Speed

• Leverages the power of XGBoost — a high-performance gradient boosting algorithm — for fast training and accurate predictions.

⚖️ Class Imbalance Handling with SMOTE

• Uses SMOTE oversampling to improve detection of rare fraudulent cases without overfitting.

🧪 F1-Optimized Threshold Tuning

Dynamically selects the best threshold using precision-recall curve to maximize the F1-score — ideal for imbalanced data.

📊 Visual Analytics Dashboard

• Histogram of fraud probabilities

• Pie chart of predicted real vs. fake listings

• Threshold vs. F1-score visualization

• SHAP summary plots for feature importance

• Explainable AI with SHAP

-Visualize top contributing features

-Interpret individual predictions

-Enhance transparency and trust in model decisions

✉️ High-Risk Job Alert System

Automatically flags listings with fraud probability ≥ 0.8 and supports email alerts via SMTP integration.

🔗 Unified Text + Numeric Feature Pipeline

Combines TF-IDF matrix with custom features using scipy.sparse.hstack for seamless model input.

⚡ Deployment-Ready with Streamlit

Integrated with a user-friendly Streamlit interface for real-time fraud prediction and insights.

🛠️ Technologies Used

Area	Tools/Libraries
Data Handling	pandas, numpy
Modeling	XGBoost, SMOTE
NLP	TfidfVectorizer
Evaluation	F1-score, precision-recall curve
Visualization	matplotlib, seaborn, SHAP
Web Dashboard	Streamlit
Deployment	Streamlit Cloud
🧾 Setup Instructions

image

• pandas and numpy: Used for handling structured data and numerical operations.

• matplotlib.pyplot and seaborn: Used for plotting graphs and visualizing distributions and model results.

• TfidfVectorizer: Converts text into numerical features based on frequency and importance.

• RandomForestClassifier: (Though unused here) A machine learning model. You used XGBClassifier instead.

• XGBClassifier: An optimized version of gradient-boosted trees, very powerful for tabular data.

• train_test_split: Splits data into training and validation sets.

• classification_report, confusion_matrix, f1_score, etc.: Used to evaluate model performance.

• SMOTE: Balances the dataset by generating synthetic examples for the minority class (fraudulent jobs).

• shap: Helps explain the model’s predictions and shows which features are most important.

• warnings.filterwarnings("ignore"): Suppresses warnings for clean output.

image

• Reading CSV files into pandas DataFrames.

• on_bad_lines='skip' ensures any corrupted lines or format errors are skipped rather than crashing the program.

image

Dataset Preparation & Feature Engineering To ensure robust and consistent model training, we preprocess and enrich the dataset through a series of thoughtful transformations:

Safe Copy of the Data • We begin by copying the input DataFrame to avoid modifying the original data. This is a best practice for all preprocessing pipelines.

Validate and Clean Text Columns • We ensure key textual columns (title, description, requirements, company_profile) exist. If missing, they're created and filled with empty strings to avoid errors during vectorization.

Clean the Email Column • If the email field exists, we replace all missing values with blank strings to maintain formatting and prevent errors during feature extraction.

Merge Text for TF-IDF

We combine all main text fields into a unified full_text column, which is used to generate TF-IDF features. This leverages the entire job listing content for better text-based learning.

Generate Custom Features

• We engineer several insightful features to detect fraud patterns:

• desc_len: Length of the job description (in characters).

 • word_count: Number of words in the description.

• num_digits_in_title: Count of numerical digits in the job title (e.g., “Earn $5000”).

• has_profile: Boolean flag — does the company provide a profile?

• suspicious_terms: Checks for keywords like "bitcoin", "transfer", "click", "money", etc., common in scams.
Email-Based Features

email_domain: Extracted from the email (e.g., gmail.com, company.com).

free_email: Flags whether the domain belongs to a free provider (like gmail, yahoo, etc.), often used in fake listings.

Consistency Across Datasets We apply the same transformation function to both training and test datasets. This guarantees consistent features during model training and inference.
image

Split the dataset:

• X: Input features from the training data.

• y: Output labels (0 = Real, 1 = Fraudulent).

• X_test: Input features from the test data (labels are not available).
Features in X:

• Text feature: Combined text column (formed by concatenating title, description, requirements, company_profile).

• Handcrafted numeric features:

• desc_len: Number of characters in the job description.

• word_count: Number of words in the description.

• num_digits_in_title: Number of digits in the job title.

• has_profile: Binary flag indicating presence of a company profile.

• suspicious_terms: Binary flag if scammy terms like money, bitcoin exist.

• free_email: Binary flag for free email domains (e.g., gmail.com).
TF-IDF Vectorization

• To handle text data, we use TF-IDF (Term Frequency-Inverse Document Frequency) — a statistical method to evaluate how important a word is in a document relative to the entire corpus

Configuration:

• max_features = 5000: Use the 5000 most informative words.

• stop_words = 'english': Removes common unimportant words (e.g., the, and, is).
Apply TF-IDF:

• On X['text'] → creates X_tfidf (sparse matrix).

• On X_test['text'] → creates X_test_tfidf (using the same vocabulary as training).
Combining All Features

The TF-IDF matrix (text features),

With 6 handcrafted numerical features (from feature engineering).

We use scipy.sparse.hstack() to horizontally stack these two sets of features into one big feature matri.

Final matrices:

• X_combined: Final feature set for training.

• X_test_combined: Final feature set for test predictions.
• This combined matrix is now ready to feed into machine learning models like XGBoost giving the model both:

These combined matrices are then used to train ML models like XGBoost or Random Forest, benefiting from:

• Rich semantic content (from text)

• Strong rule-based signals (from numeric features
image

• Handling Class Imbalance and Model Training

Problem: Class Imbalance

• Fraudulent job postings are rare.

• This causes class imbalance — a major challenge in machine learning.

• A model that always predicts “not fraud” may achieve high accuracy but is useless.
Solution: SMOTE (Synthetic Minority Oversampling Technique)

• SMOTE creates synthetic fraud examples by interpolating between real ones.

• This balances the dataset and helps the model learn fraud patterns more effectively.
Train-Validation Split

• Dataset is split using train_test_split() with stratify=y to maintain class distribution.
Model: XGBoost Classifier

• Chosen for its speed and accuracy in classification tasks.
• Key hyperparameters:

 • n_estimators=200: Number of trees.

 • max_depth=6: Controls model complexity to avoid overfitting.

 • subsample & colsample_bytree: Randomly sample rows and features to enhance generalization.

 • eval_metric='logloss': Metric to evaluate binary classification error.
Training

• The model is trained on SMOTE-balanced data: X_res, y_res.

image

• Threshold Optimization for Better Performance

Default Threshold Issue

• Most models use a default threshold of 0.5 for binary classification.

 • On imbalanced datasets, this may not yield optimal results.
Threshold Tuning

• Use precision_recall_curve to compute scores across all possible thresholds.

• Calculate F1 Score (harmonic mean of precision and recall) for each threshold.

• Select the best_threshold that maximizes F1 Score.
Plot F1 vs Threshold

• Helps visualize how F1 score changes with different classification thresholds.

• Aids in selecting the optimal best_threshold.
Final Predictions with Best Threshold

• Apply best_threshold to the model's output probabilities.

• Generate predictions for evaluation.
Evaluate with classification_report()

• Provides detailed performance metrics:

• Precision: Of all predicted frauds, how many were actually fraud?

• Recall: Of all actual frauds, how many did the model detect?

• F1 Score: Harmonic mean of precision and recall — balances false positives and false negatives.

• Support: Number of true instances in each class.
Evaluate with confusion_matrix()
• Breaks down predictions into four categories:

    • True Positives (TP) – Correctly predicted fraud cases.

    • True Negatives (TN) – Correctly predicted genuine jobs.

    • False Positives (FP) – Genuine jobs incorrectly labeled as fraud.

    • False Negatives (FN) – Fraud cases the model failed to detect
• This helps us understand model performance beyond accuracy and ensures it catches fraud with minimal false alarms.

image

Now that our model is trained and fine-tuned, we use it to predict the fraud probability of each job listing in the test dataset.

Use the Trained Model to Predict Fraud Probabilities:

 • model.predict_proba(X_test_combined)[:, 1]
 → Returns the probability of class 1 (fraud) for each job listing in the test set.
Store Probabilities:

• Save the output as a new column:
   test_df['fraud_probability']
Convert Probabilities to Binary Predictions:

• Apply the previously selected best_threshold:

 • If probability ≥ threshold → fraud_predicted = 1 (high likelihood of fraud)

 • If probability < threshold → fraud_predicted = 0 (likely genuine)
• Save this as: test_df['fraud_predicted']

Result:
• A clean column with 0s and 1s representing model predictions for fraud.

📊 Visualizing Model Confidence 5. Histogram of Fraud Probabilities:

  • Plot all values in test_df['fraud_probability']

  • Peaks near 0 → Model is confident many jobs are real.

  • Peaks near 1 → Model has high confidence in fraud predictions.
KDE Line (Smooth Curve):

• Added to the histogram to better visualize the shape of the probability distribution.
• Visualizing Fraud vs Real Predictions

Pie Chart of Predicted Labels:

• Use values from test_df['fraud_predicted'] to count real vs fake predictions.

• Provides a clear view of the percentage of job listings flagged as fake.
• Helps decision-makers or investigators understand how widespread fraud is in the test dataset according to the model.

image

• Why SHAP?

• SHAP (SHapley Additive exPlanations) is a method from game theory used to explain individual predictions made by machine learning models. It assigns each feature a contribution score (positive or negative) toward the final prediction.

Dense Conversion for SHAP Compatibility

• SHAP requires input data in dense format.
→ We convert a subset of the training data (e.g., X_res[:100]) using .toarray().
SHAP Explainer Setup

• We create a SHAP explainer using the trained XGBoost model and the dense data.
→ This explainer understands the model logic and computes contribution scores.
SHAP Values Computation

• SHAP values tell how much each feature pushes a prediction toward class 1 (fraud) or class 0 (non-fraud).

Feature Names
• We combine:

     • Top 5000 TF-IDF features from job descriptions and titles

     • Custom-engineered features like:

     • desc_len

     • word_count

     • num_digits_in_title

     • has_profile

     • suspicious_terms

     • free_email
Summary Plot Visualization
• We generate a SHAP summary plot which shows:

• Most influential features

• Direction (does a feature increase or decrease fraud risk?)

• Distribution and strength of these features across samp
image

High-Risk Job Listings Alert

• While your model predicts all fraud probabilities, not every “fraudulent” prediction has the same confidence. A job predicted at 0.51 might just barely cross the threshold — but one predicted at 0.95 is likely very suspiciou

Why This Matters
• Jobs with very high fraud probability (e.g., 0.95) are more likely to be genuinely fraudulent and may require:

 • Human verification or moderation

 • Platform filtering/blocking

 • Alerts to internal teams
Implementation Overview
• Thresholding A cut-off of 0.80 is used to flag high-confidence frauds. (This value can be adjusted based on acceptable risk tolerance.)

• Filtering From the test dataset, all listings with fraud_probability ≥ 0.8 are filtered into a separate DataFrame.

• Formatted Alert For each high-risk listing, an alert message is generated with:

• Title

• Location

• Fraud Probability (rounded to 2 decimal places)
📨 OUTPUT BASED ON SAMPLE INPUT

image

F1 Score Optimization

The F1 Score combines Precision and Recall into a single metric, making it ideal for imbalanced datasets like fraud detection where accuracy alone can be misleading. image

Threshold Analysis

• X-axis: Probability thresholds for classification (from 0 to 1)

• Y-axis: F1 Score achieved at each threshold

• A blue curve shows how the F1 Score changes across different thresholds. • A red dashed vertical line marks the best-performing threshold.

Key Results • Best Threshold: 0.21

    • Instead of the standard 0.5 threshold, our model performs best when classifying any instance with a fraud probability above 21% as fraudulent.
• Peak F1 Score: ~0.81

    • This is the highest achievable balance between precision and recall with our current model.l
Why such a low threshold?

• our model tends to predict conservative (low) probabilities even for actual fraud cases, so using a lower cutoff helps catch more fraudulent transactions while maintaining reasonable precision
Implementation Impact

• This threshold optimization significantly improves model performance by:

  • Increasing Recall - Catching more actual fraud cases

  • Maintaining Precision - Avoiding excessive false positives

  • Maximizing F1 Score - Achieving optimal balance for our imbalanced dataset
image

• Threshold Tuning

• To optimize fraud detection, we tuned the classification threshold instead of using the default 0.5. The best performance was achieved at threshold = 0.21, giving the highest F1 Score of 0.8109 — effectively balancing precision and recall.

 Threshold	         F1 Score
  0.10	           0.7230
  0.21	           0.8109 
0.25–0.50	         ~0.78–0.79
• Final Performance (Validation Set)

 Metric                   Value                     Interpretation                                    

• Accuracy                 0.98                        Overall correct predictions.                      
• Precision (Fraud)        0.78                        78% of fraud predictions were correct.            
• Recall (Fraud)           0.85                        85% of actual frauds were detected.               
• F1 Score (Fraud)         0.81                        Balanced metric of fraud detection.               
• Macro Avg (F1)           0.90                        Treats both classes equally.                      
• Weighted Avg (F1         0.98                        Reflects performance considering class imbalance. 
• Confusion Matrix

                          Predicted Real	                 Predicted Fraud
• Actually Real	         2688 	                          34 (False Positives)
• Actually Fraud	        21 (False Negatives)	           118 
• Key Insights

• Only 21 fraud cases were missed (false negatives).

• Only 34 real jobs were wrongly flagged as fraud (false positives).
• Most classifications are correct — this is a well-performing, balanced model.

image

Key Insights from the Fraud Probability Distribution

Distribution Pattern:
What This Graph Tells Us

• X-axis: Fraud Probability The probability output by your model that a job posting is fraudulent (ranges from 0 to 1).

Closer to 0 = more likely real Closer to 1 = more likely fraud

• Y-axis: Frequency How many samples fall into each probability range/bin

• Most predictions clustered near 0 (legitimate jobs)

• Small cluster near 1.0 (fraudulent jobs)

• Sparse middle values indicate decisive model behavior
Model Behavior:

• High confidence predictions - model rarely assigns intermediate probabilities (0.4-0.6)

• Strong feature separability - clear distinction between fraud and legitimate postings

•  The pattern reflects a class imbalance—fraudulent jobs are rare compared to legitimate ones.

• A smoothed KDE curve (red line) highlights the underlying distribution, confirming the two peaks (at 0 and 1).
Practical Implications:

• Low threshold justified - setting threshold around 0.21 captures frauds in the right tail

• High threshold problematic - using 0.5+ would miss many actual frauds

• The model is reliable for the majority class (legitimate jobs), and also identifies true frauds at the far right.
image

What This Pie Chart Shows

Title: “Predicted Fake vs Real Job Listings”

Prediction Breakdown:

• 94.8% of job listings are predicted as real (light blue).

• 5.2% are predicted as fraudulent (salmon red).
Class Imbalance:

• The chart reflects real-world data where most jobs are genuine.
Fraud Detection Strategy:

• 5.2% is significantly higher than the true base rate of fraud in most datasets (usually <2%), meaning your model is trying to err on the side of caution, which is a good strategy in fraud detection.
Practical Implication:

• It’s likely catching more fraud at the cost of a few false positives, which is usually acceptable in fraud screening systems.
• This cautious approach is typical and effective in fraud detection systems
image

Feature Importance Order:

• Features are listed top-to-bottom by their average impact on predictions (most important at the top).
SHAP Value (X-axis):

• Positive SHAP values: Push prediction toward fraud (class = 1).

• Negative SHAP values: Push prediction toward real (class = 0).
Dots (Individual Predictions):

• Each dot represents one job listing.

• Dot position shows how much that feature influenced the fraud prediction for that listing.
Color (Feature Value):

• Red/pink = High feature value.

• Blue = Low feature value.
Key Feature Insights:

• has_profile:

       • Most influential feature.

       • High value (red, likely “no profile”) strongly pushes toward fraud.

       • Low value (blue, “has profile”) pushes toward real.

       • word_count & desc_len:

       • High values (red, longer descriptions) push toward real jobs.

       • Low values (blue, short descriptions) push toward fraud.
• Keyword Features (e.g., fun, goal, team, years):

  • Words like “fun” and “goal” (red) tend to push toward fraud—possibly because fake jobs use vague, motivational language.

  •  Words like “team,” “years,” and “solutions” push toward real—suggesting structured, corporate language signals legitimacy.
🚀 Live Demo

• Click here to open the Streamlit App

• Features

  • Fraud Detection  using XGBoost, TF-IDF, and SMOTE
  • Interactive UI Design built with Streamlit
  • Custom Threshold Slider to tune sensitivity
  • Live Metrics Dashboard with precision, recall, F1-score, and confusion matrix
  • Search & Filter functionality for job posts
  • Visualizations Pie charts and histograms for data insights
  • Downloadable Results in CSV format
THIS IS THE Also the link:-

https://fraud-detector-app-cgr9btenw5tzsp72fpv2gu.streamlit.app/

photos of app we have designed

UI DESIGN OF THE APP

image

output after browsing the file

image

image

📄 fraud_app (12).py
This script contains the complete Streamlit code for deploying the fraud detection app, including model loading, threshold tuning, performance metrics, and interactive UI components.

📔 Colab Notebook
fraud_app.ipynb you can excess the code from here

This notebook includes model training with TF-IDF, SMOTE, XGBoost, and evaluation before deployment.

since we have to speed up the video in order to compressed to 8mins.so we kindly request to pls watch this video at speed 0.5x for better viewing experience.pls consider our request as we have put enormous amount of work into completion of this project

🎥 [Watch the presentation video]

— Recommended speed: 0.5x for the best viewing experience.

About
🔍 Job Fraud Detector | Anveshan Hackathon Project An ML-powered Streamlit web app that detects fake job listings using natural language processing, feature engineering, and XGBoost. Includes real-time predictions, SHAP explanations, search, filtering, and CSV export — built to protect job seekers from online scams

Resources
 Readme
 Activity
Stars
 0 stars
Watchers
 0 watching
Forks
 0 forks
Report repository
Releases
No releases published
Packages
No packages published
Languages
Jupyter Notebook
93.7%
 
Python
6.3%
Footer
