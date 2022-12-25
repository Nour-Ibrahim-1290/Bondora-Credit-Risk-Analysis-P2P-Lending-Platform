
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
#### To display all the columns of dataframe

```bash
pd.set_option('display.max_columns', 500)
import warnings
warnings.filterwarnings("ignore")
``

#### Usage

```google.colab
from google.colab import drive
drive.mount('/content/drive')

Mounted at /content/drive
```

#### Load Dataset
```bash
df = pd.read_csv('/content/drive/MyDrive/Technocolabs_Team/Bondora_preprocessed.csv')
```
#### Data Preprocessing
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
#### Contributing

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

#### Task Performed by: 
all team members made their own version and with the help of our mentor, we selected a suitable apprach and used it for later stages of analysis.
Reported By: Babli

## 3. Exploratory Data Analysis

**Univariate Analysis:**

Plotted histogram to see the distribution of data for each column and found that few variables are normally distributed and most of the variables are Right Skewed.

**Correlation Plot of Numerical Variables:**

Positive Correlations:
1. PrincipleBalance, and [Amount, AppliedAmount]
2. MonthlyPayment, and [Amount, AppliedAmount]
3. NoOfPreviousLoansBeforeLoan, and [AmountOfPreviousLoansBeforeLoan, ExisitingLiabilities]
4. MonthlyPayment, and PrincipleBalance
5. BidsPortofolioManager, and [PrinciplePaymentsMade, InterestAndPenaltyAndPaymentsMade]
\
Negative Correlations:\
There's no severe negative correlation.

Removed the independent variables having high correlation with each other.

**Visualisation of Variables:**

By visualizing variables using different graphs such as Barplot, Histograms, Piechart etc. come with some insights:
- There are 56.3% of the loans are Defaulted and 44.7% are Not Defaulted.
- The defaulters are more in number who had their Secondary education and followed by Higher education and Vocational education.
- The borrowers who had their employment status as Not Present are defaulted, followed by Fully Employed.
- Borrowers having rating ‘F’ are defaulted followed by rating as ‘HR’.
- Borrowers with credit score EsMicroL ‘M’ are defaulted.
- Mean age of borrowers is 40 years.

#### Task Performed by: 
all team members made their own version and with the help of our mentor, we selected a suitable apprach and used it for later stages of analysis.
Reported By: Nikhil


## 4. Feature Engineering
In Feature Engineering we did following task 
1. Handling Null values :-Removing all the features which have more than 40% missing values, & in rest of colums replaced null value by "Not specified " or "Not preset "

2. Handling outliers:- We checked outliers by using scatter plot ,Box plot,Z score,IQR . We handled the outliers for each numerical feature using "Winsorizing method".
                       
3. Feature Selection:- we used correlation filter selection technique which means Highly correlated features will be considered duplicated features while using the machine learning model,
                      so we should drop them(drop one and leave the another)
4. Categorical Features Encoding:- we divided our feaytures to "Target" & "Independent " Features

5. Feature scaling:-There are two major types of feature scaling : A)Standardization B)Normalization.
                   We used StandardScalar to scale our data: StandardScaler is used to resize the distribution of values ​​so that the mean of the observed values ​​is 0 and the standard deviation is 1.
                   The values will lie be between -1 and 1.
 
6. Feature extraction and Dimensionality reduction using PCA:- We used PCA to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information.

7. Spiliting Data into train and test sets:- we split data into train & test set & made data ready for MLA
                                            For splitting data we used Train Test split & K-Fold Cross-Validation
                                            
#### Task Performed by: 
all team members made their own version and with the help of our mentor, we selected a suitable apprach and used it for later stages of analysis.
Reported By: Aishwani, and Abdul


## 5. Classification Modeling (Probability of Default)

#### Metrics considered for Model Evaluation
**Accuracy , Precision , Recall and F1 Score**
- Accuracy: What proportion of actual positives and negatives is correctly classified?
- Precision: What proportion of predicted positives are truly positive ?
- Recall: What proportion of actual positives is correctly classified ?
- F1 Score : Harmonic mean of Precision and Recall

#### Gradient Boosting Classifier

Gradient boosting classifiers are a group of machine learning algorithms that combine many weak learning models together to create a strong predictive model. Decision trees are usually used when doing gradient boosting.

Training Gradient Boosting classifier:
Computed the accuracy scores on train and validation sets when training with different learning rates. When learning rate was 0.5, the accuracy scores on training and validation subsets were 0.84 and 0.830, respectively.
Trained Gradient Boosting classifier on training subset with parameters of criterion="mse", n_estimators=50, learning_rate = 0.5, max_features=2, max_depth = 5, random_state = 0. The average precision, recall, and f1-scores on validation subsets were 0.840, 0.85, and 0.85, respectively.

Training Gradient Boosting classifier with Hyperparameter Tunning:
Hyperparameter Tuning using RandomizedSearchCV:
GradientBoostingClassifier(*, loss='deviance', learning_rate=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90], n_estimators=50, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=5, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)

Best Estimators achieved using GBC on PCA Dataset Accuracy: ~ 85%
learning rate of 0.3
n_estimators 50
max_depth 5

#### Random Forest Classifier

Random forest is a supervised learning algorithm. It has two variations – one is used for classification problems and other is used for regression problems. It is one of the most flexible and easy to use algorithm. It creates decision trees on the given data samples, gets prediction from each tree and selects the best solution by means of voting. It is also a pretty good indicator of feature importance.

Random forest algorithm combines multiple decision-trees, resulting in a forest of trees, hence the name Random Forest. In the random forest classifier, the higher the number of trees in the forest results in higher accuracy.

**Defining base-model with default parameters**

'''from sklearn.ensemble import RandomForestClassifier'''
**accuracy score: 0.846**

#### Defining a Random Forest Classifier using Hyperparameter tunnimg

'''from sklearn.model_selection import RandomizedSearchCV'''

Trained Random Forest Classifier on training subset with parameters of criterion="gini", "entropy", n_estimators=list(range(10,200)), max_features= list(range(10, X_test.shape[1]+1)), max_depth = [5, 10, 15]. The average precision, recall, and f1-scores on validation subsets were 0.840, 0.85, and 0.84, respectively.
**roc_auc_score =  0.9223306591778067**

Conclusion:

Random Forest Classifier, with parameters of:
max_depth=15,
max_features=22,
min_samples_leaf=4,
n_estimators=115
Achieved the highest performance in classifying the Defaulted and Non-defaulted Loans.

#### Task Performed by: 
this task was performed by 2 teams, each one on it's own, by: Nour M. Ibrahim, Subhadeep, Abdul, Aishwani, Ritu
Reported By: Suhadeep

## 5. Assesment Target Variables Creation

#### Task Performed by: 
Nour M. Ibrahim
Reported By: Nour M. Ibrahim


## 6. Regression Modeling
#### Task Performed by: 
This task was performed by 2 teams, each one on it's own, by: Nour M. Ibrahim, Prashaant, Aishwani, Ritu, Ahmed
Reported By: Prashaant


## 7. Pipelines Creation
#### Task Performed by: 
Nour M. Ibrahim, Simran Katyar
Reported By: Nour M. Ibrahim

## 8. Model Deployment

#### Task Performed by: 
Nour M. Ibrahim, Simran Katyar
Reported By: Nour M. Ibrahim
