
# Credit Risk Analysis for Bondora online P2P Platform

This project was dedicated to the analysis, modeling, and assesing possible risks around Credits happening on [**Bondora**](https://www.bondora.com/) platform for P2P credits investments.

### In which we have gone through this develping cycle:
1. Data Collection (directly form the platform's website as a csv file)
2. Data Preprocessing.
3. Explaratory Data Analysis.
4. Feature Engineering
5. Classification Modeling (Probability of Default).
6. Target variable creation for risk evaluation and assesment.
7. Regression Modeling.
8. Pipelines Creation (Classification and Regression).
9. Model Deployment for production (App Creation & AWS EC2 deployment).


## Team Memebers:
- Yasin Shah (mentor)
- Nour M. Ibrahim (team-lead)
- Simran Katyara.
- Aishwani 
- Ritu
- Babli 
- Suhadeep
- Abdul
- Ahmed
- Nikhil
- 

## 1. Data Collection

In this project we will be doing credit risk modelling of peer to peer lending Bondora systems.Data for the study has been retrieved from a publicly available data set of a leading European P2P lending platform  ([**Bondora**](https://www.bondora.com/en/public-reports#dataset-file-format)).The retrieved data is a pool of both defaulted and non-defaulted loans from the time period between **1st March 2009** and **27th January 2020**. The data
comprises of demographic and financial information of borrowers, and loan transactions.In P2P lending, loans are typically uncollateralized and lenders seek higher returns as a compensation for the financial risk they take. In addition, they need to make decisions under information asymmetry that works in favor of the borrowers. In order to make rational decisions, lenders want to minimize the risk of default of each lending decision, and realize the return that compensates for the risk.

## 2. Data Preprocessing
```bash
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for visualization
import plotly.express as px # for visualization
import matplotlib.pyplot as plt # for visualization
%matplotlib inline
```
# To display all the columns of dataframe

```bash
pd.set_option('display.max_columns', 500)
import warnings
warnings.filterwarnings("ignore")
``

## Usage

```google.colab
from google.colab import drive
drive.mount('/content/drive')

Mounted at /content/drive
```

## Load Dataset
```bash
df = pd.read_csv('/content/drive/MyDrive/Technocolabs_Team/Bondora_preprocessed.csv')
```
## Data Preprocessing
The dataset contains **112** Columns and  **134529** Rows Range Index 
```bash
Index(['ReportAsOfEOD', 'LoanId', 'LoanNumber', 'ListedOnUTC',
       'BiddingStartedOn', 'BidsPortfolioManager', 'BidsApi', 'BidsManual',
       'UserName', 'NewCreditCustomer',
       ...
       'PreviousEarlyRepaymentsCountBeforeLoan', 'GracePeriodStart',
       'GracePeriodEnd', 'NextPaymentDate', 'NextPaymentNr',
       'NrOfScheduledPayments', 'ReScheduledOn', 'PrincipalDebtServicingCost',
       'InterestAndPenaltyDebtServicingCost', 'ActiveLateLastPaymentCategory'],
      dtype='object', length=112)

RangeIndex(start=0, stop=134529, step=1)
```
## Contributing

Removing the columns having missing value for then 40% missing values . after remove null columns now we have **48** columns and **57135** Rows for further use.

##Checking Distribution of Categorical Variable
```bash
for col in cat_cols:
  print(f'{col} has {df[col].nunique()} unique values:')
  print(df[col].unique())
  print('*'*40)```

We can see that: There're date attributes in categorical form --> Identify and drop them, as they're not relevant to our analysis.

##checking distribution of Numerical columns
```bash
for col in num_cols:
  print(f'{col} has {df[col].nunique()} unique values:')
  print(df[col].unique())
  print('*'*40)```

*We can see in numeric column distribution there are many columns which are present as numeric but they are actually categorical as per data description such as Verification Type, Language Code, Gender, Use of Loan, Education, Marital Status,EmployementStatus, OccupationArea etc. --> So we will convert these features to categorical features.
*Also, there're columns that has values out of its ecoding range, like MatrialStatus, Education, LanguageCode, etc. --> Handle these to follow under the specified encoding.

##VerificationType
```bash
#counts of each status categories df['Status'].value_counts()
```
 * Gender
 * LanguaugeCode
 * UseOfLoan
 * Education
 * MaritalStatus
 * Employmentstatus
 * newCreditCustome
 * Restructred
 * OccupationArea
 * HomeOwnershipType


As we can see from above stats most of the loans are -1 category whose description is not avaialble in Bondoro website so we have dig deeper to find that in Bondora most of the loans happened for which purpose so we find in Bondora Statistics Page most of the loans around 34.81% are for Not set purpose. so we will encode -1 as Not set category.

## 3. Exploratory Data Analysis
