# 🛡️ FraudShield – Credit Card Fraud Detection (Web App : https://fraud-detection-app-626338298812.europe-west1.run.app/)

FraudShield is a machine learning project I developed as part of my MSc Data Science dissertation. The main aim of the project was to compare different machine learning approaches for detecting fraudulent credit card transactions and also build a simple web application to demonstrate the predictions.

I used the IEEE-CIS Fraud Detection dataset, which contains more than 590,000 transactions. After combining the transaction and identity datasets, the final dataset had 435 features. The dataset was highly imbalanced, with only around 3.5% of the transactions marked as fraud. :chatgpt-content-reference{index="0"} :chatgpt-content-reference{index="1"}

## Models Used

I trained and compared nine different machine learning models:

- Logistic Regression
- Random Forest
- XGBoost
- CatBoost
- Multilayer Perceptron (MLP)
- Isolation Forest
- Autoencoder
- Soft Voting
- Stacking

These models were chosen to compare supervised learning, anomaly detection and ensemble learning approaches.

## Data Processing

Before training the models, I carried out exploratory data analysis and preprocessing.

Some of the main steps included:

- analysing missing values
- handling categorical and numerical features
- frequency encoding categorical variables
- log transforming transaction amount
- creating `RelativeHour` and `RelativeWeekday` features
- removing highly missing and correlated features
- scaling features for models that required it

The transactions were split chronologically into 60% training, 20% validation and 20% test data. I used a time-based split instead of a random split so that the models were trained on earlier transactions and tested on later transactions, which is closer to a real fraud detection scenario. :chatgpt-content-reference{index="2"}

## Model Evaluation

Because fraud transactions were a very small part of the dataset, I used **PR-AUC as the main evaluation metric**.

I also compared the models using:

- ROC-AUC
- Precision
- Recall
- F1-score
- Accuracy
- Confusion Matrix

The decision threshold for each model was selected using the validation dataset instead of simply using the default threshold of 0.5.

I also used bootstrap sampling to calculate 95% confidence intervals for PR-AUC and ROC-AUC and checked the temporal stability of the best-performing model.

## Results

The **Stacking model gave the highest PR-AUC of 0.4847**, followed by Soft Voting and the tree-based models such as XGBoost, CatBoost and Random Forest.

However, the confidence intervals of the top-performing models overlapped, so the results did not show that Stacking was clearly better than every other model. :chatgpt-content-reference{index="3"}

The results also showed that tree-based and ensemble models worked better on this dataset compared with Logistic Regression, MLP and the anomaly-detection models. :chatgpt-content-reference{index="4"}

## Explainable AI

I used **SHAP** to understand why the models were making their predictions.

For XGBoost, I analysed both overall feature importance and individual transaction predictions. Some important features included `C14`, `V258`, `card6` and `TransactionAmt_log`.

One interesting result was that the model was not depending on one single feature. Instead, the prediction was based on the combined effect of many different features. :chatgpt-content-reference{index="5"}

## FraudShield Web Application

I also developed a Flask-based web application called **FraudShield** to show how the trained models could be used in a simple application.

The application includes:

- live transaction monitoring
- fraud probability prediction
- manual transaction checking
- switching between different models
- model performance visualisations
- SHAP explanations
- API demonstration

For the live monitoring feature, I used **Server-Sent Events (SSE)** to replay transactions from the test dataset one by one and simulate a real-time transaction stream.

The web application was created as a research prototype rather than a production banking system. :chatgpt-content-reference{index="6"}

## Technologies Used

Python, Pandas, NumPy, Scikit-learn, XGBoost, CatBoost, TensorFlow/Keras, SHAP, Flask, Joblib, JavaScript, HTML and CSS.


