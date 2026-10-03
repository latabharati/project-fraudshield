# Project Structure

The main files and folders used in the FraudShield project are organised as follows:

```text
FraudShield/
│
├── artifacts/
│   └── Saved trained models and preprocessing files used by the application
│
├── catboost_info/
│   └── Files automatically generated while training CatBoost models
│
├── data/
│   └── Processed datasets and data files used by the project
│
├── static/
│   └── CSS, JavaScript and other static files used by the Flask application
│
├── templates/
│   └── HTML templates used for the FraudShield web application
│
├── FraudDetection-EDA.ipynb
│   └── Exploratory Data Analysis of the IEEE-CIS Fraud Detection dataset
│
├── FraudDetection-Preprocessing...
│   └── Data preprocessing, feature engineering and model preparation
│
├── app.py
│   └── Main Flask application used to run FraudShield
│
├── requirements.txt
│   └── Python libraries required to run the project
│
├── Procfile
│   └── Configuration used for deploying the Flask application
│
├── .python-version
│   └── Python version used by the project
│
├── .gitignore
│   └── Files and folders excluded from Git
│
├── .gitattributes
│   └── Git configuration for repository files
│
├── README.md
│   └── Overview of the FraudShield project
│
└── project_structure.md
    └── Description of the repository structure

Main Components
Data Analysis
FraudDetection-EDA.ipynb contains the exploratory data analysis carried out before model development, including missing-value analysis, fraud distribution and relationships between different transaction features.
Preprocessing and Modelling
The preprocessing notebook contains the main steps used to prepare the IEEE-CIS dataset for machine learning, including missing-value handling, categorical encoding, feature engineering and preparation of the train, validation and test datasets.
Model Artifacts
The artifacts folder contains trained models and preprocessing objects saved during the machine learning process. These files are loaded by the Flask application when predictions are made.
Web Application
The main application is contained in app.py.
The templates folder contains the HTML pages, while the static folder contains the CSS and JavaScript files used for the user interface.
Deployment
requirements.txt, Procfile and .python-version contain the main configuration needed to run and deploy the application.
