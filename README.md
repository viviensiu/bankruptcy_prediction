# backruptcy_prediction
Background: The UCI ML repository has a Taiwanese Bankruptcy Prediction data set which were collected from the Taiwan Economic Journal for the years 1999 to 2009. Company bankruptcy was defined based on the business regulations of the Taiwan Stock Exchange.

Source: https://archive.ics.uci.edu/ml/datasets/Taiwanese+Bankruptcy+Prediction

Objective: We will use this data set to train a ML model that predicts company bankruptcies in Taiwan. 

# Methodology:
1) Perform EDA on the 6 basic financial ratios for normal and bankrupt companies.
2) Perform statistical analysis using Mann-Whitney Test on the following financial ratios for normal and bankrupt companies: Debt Ratio, Current Ratio, Quick Ratio, Net Profit Margin before Tax, Current Liability To Assets, Net Income To Stockholder's Equity.
3) Perform Chi-Square Contingency Test on categorical feature 'One If Total Liabilities Exceeds Total Assets Zero Otherwise' for normal and bankrupt companies.
4) Upsample the training data set with 4 different upsampling algorithms: SMOTE, ADASYN, SMOTEENN, SMOTETomek. For each upsampled data:
    (i) Train and predict the dataset using the following models: 
        - DummyClassifier (as baseline), 
        - LogisticRegression, 
        - RandomForestClassifier, 
        - GradientBoostingClassifer. 
    (ii) For each model, evaluate using F1 scores, confusion matrix and classification report.
5) From step 4, pick the top three best combination of upsampler-model by looking at confusion matrix and F1 scores.    
6) Find the best parameters for the best models using hyperparameter tuning.
7) Train and predict using models with best parameters.
8) Evaluate models from Step 7 using F1 scores, confusion matrix and classification report.

Technical Skills: 
- Pandas, Scipy, Seaborn, Matplotlib, statistics test (Mann-Whitney U Test, Chi-Square Contingency Test), Upsampling (imbalanced-learn, SMOTE, ADASYN, SMOTEENN, SMOTETomek), hyperparameter tuning (GridSearchCV), Machine Learning classification modelling. 

# Conclusion:
Out of 4 ML models, Logistic Regression produce the lowest false negative scores of only 7 using SMOTEENN upsampler, however, the F1 score is 0.291339 due to the higher number of false positives (173).

This is followed by GradientBoostingClassifier with second lowest false negative scores of 12 using SMOTEENN upsampler. Its F1 score is 0.397516 and 85 false positives.

The RandomForestClassifier model (upsampled with SMOTEENN) has the best F1 score out of all models: 0.439716, a false negative score of 13 and 66 false positives.

We hereby conclude that, if the model is used for the sole purpose of reducing the risks of financial loans default by lenders, the Logistic Regression model would be the ideal choice. However its high false positive rates imply a higher number of false alarms when it comes to reviewing a borrower's financial portfolio.

If a financial company is expanding its loan business and the model is to be used as a supplementary risk assessment for loan approvals, the GradientBoostingClassifier model or RandomForestClassifier model would be a better choice due to a much lower number of loan applications that would be flagged wrongly as potentially bankrupt borrowers. 

# Ideas for expanding this work:
- The same methods can be applied onto similar problem statement such as predicting credit card defaults and personal loan defaults.

# Useful Reading:
1) 6 Basic Financial Ratios (https://www.investopedia.com/financial-edge/0910/6-basic-financial-ratios-and-what-they-tell-you.aspx)
2) Non-parametric tests: Sign test, Wilcoxon signed rank, Mann-Whitney U test (https://www.youtube.com/watch?v=IcLSKko2tsg)
3) Mann-Whitney U test (https://datatab.net/tutorial/mann-whitney-u-test)
4) Mann-Whitney U test in Python (https://www.reneshbedre.com/blog/mann-whitney-u-test.html)
5) A Gentle Introduction to the Chi-Squared Test for Machine Learning (https://machinelearningmastery.com/chi-squared-test-for-machine-learning/)
6) Scikit-Learn: Machine learning in Python (https://scipy-lectures.org/packages/scikit-learn/index.html)
7) SMOTE for imbalanced data (https://www.kite.com/blog/python/smote-python-imbalanced-learn-for-oversampling/)
8) ADASYN: Adaptive Synthetic Sampling Method for Imbalanced Data (https://towardsdatascience.com/adasyn-adaptive-synthetic-sampling-method-for-imbalanced-data-602a3673ba16)
9) Tune Hyperparameters with GridSearchCV (https://www.analyticsvidhya.com/blog/2021/06/tune-hyperparameters-with-gridsearchcv/)
