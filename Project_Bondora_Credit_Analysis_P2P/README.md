# Credit Risk Analysis for Bondora online P2P Platform

### Project Team

Mentor: [Yasin Shah](https://github.com/Technocolabs100) 

Team Lead: [Nour M. Ibrahim](https://github.com/Nour-Ibrahim-1290)

Team members:
1. [Simran Katyar](https://github.com/SimranKatyar)
2. [Babli Wadhwa](https://github.com/BabliWadhwa)
3. [Ashwini Sanjay](https://github.com/AshwiniKakulte)
4. [Subhadeep Paul](https://github.com/Subhadeepgithib10)
5. [Ritu Mahali](https://github.com/ritumahali96)
6. [Prashant Kumar](https://github.com/kumarprashant0797)
7. [Umair Azad](https://github.com/Umair-Azad)
8. [Abdul Rafay](https://github.com/ABDULRAFAY757)
9. [Ahmed Sabry](https://github.com/Foxdanger1412)
10.[Nikhil Navnath](https://github.com/nikhilpokharkar)
11.[Akshay Khot](https://github.com/AKSHA1498)

Content:
---------
1. Requirenment & Analysis
2. Project Planning
3. Design
4. Development (Coding & Implementation)
5. Deployment
6. Conclusion

![SDLC](https://user-images.githubusercontent.com/93732090/209706167-09404f7c-ef4f-47fc-8a54-3771bee17f5b.png)

## 1. Reuirenment & Analysis
1.1 Introduction

1.2 About Data

1.3 Tools and Technologies

### 1.1 Introduction
- [**Bondora**](https://www.bondora.com/) is investment platform, which connects between investors and investees which includes Peer-to-peer lending.
- Peer-to-peer lending has attracted considerable attention in recent years, largely because it offers a novel way of connecting borrowers and lenders. But as with other innovative approaches to doing business, there is more to it than that.
- The reauiremnets for this project is to:
      1- credit risk assesmnet of new borrowers.
      2- Probability of Default of new borrowers.

### 1.2 About Data
- Data for the study has been retrieved from a publicly available data set of a leading European P2P lending platform  ([**Bondora**](https://www.bondora.com/en/public-reports#dataset-file-format)).
- The retrieved data is a pool of both defaulted and non-defaulted loans from the time period between **1st March 2009** and **27th January 2020**.
- The data
comprises of demographic and financial information of borrowers, and loan transactions.In P2P lending, loans are typically uncollateralized and lenders seek higher returns as a compensation for the financial risk they take.
- In addition, they need to make decisions under information asymmetry that works in favor of the borrowers.
- In order to make rational decisions, lenders want to minimize the risk of default of each lending decision, and realize the return that compensates for the risk.

### 1.3 Tools & Technologies
- During this project we're using Python for coding.
- Pandas module for Data Preprocessing
- Pandas, numpy, matlpotlib.pyplot, and seaborn for EDA.
- Pandas, numpy, PCA for Feature Engineering.
- Pandas, numpy, sklearn, pickle for Modeling, and Pipelining
- flask, HTML5, CSS for App development.
- AWS - EC2 for Deployment.

## 2. Project Planning
After Careful analysis of project requirenments, and the different attribtes defined on the dataset;
The project was planned to have the following Outputs:
1. Automated Assesment of...
      - Probability of Default (PD)
      - Eligible Loan Amount (ELA)
      - Equity Monthly Installemnets (EMI)
      - Preferred Return On Investment (PROI)
2. Web app for the comapny to use it for the assesment process...  
      - Enter a borrower assesment data maually --- v01
      - Upload a csv file of several borrowers and assess them at the same time ---- v02
      - Automated assesment for the latest added borrowers through Bondora's API ---- v03

## 3. Design
After careful examination of the data set, we decided to have thses Design Attributed:
**High Level Design**
1. a Classofication Pipeline to asses the Probability of Default.
2. Based on domain reaserch and weight of evidence techniaues, define 3 new algorithms to calculate EMA, ELA, and PROI.
3. a MultiRegression Pipeline to asses all three new defined attributes.
4. A Web App as stated in the requirenments by the Client, yet the set of attributes to be defined after throughtful analysis of the attributes provided of the dataset.

![Full Design - High Level](https://user-images.githubusercontent.com/93732090/209706230-5532394b-36c2-49af-85fb-b2a2531f7609.png)

**Low Level Design**

![Pipelining_flowchart](https://user-images.githubusercontent.com/93732090/209706272-0560ca16-b08d-4a58-beb3-113560baffe6.png)


![AppCreation-flowchart](https://user-images.githubusercontent.com/93732090/209706284-59ee341a-4f5d-4c92-a1be-60dc1bb04ed1.png)


## 4. Development (Coding & Implementation)

4.1. Data Preprocessing.

4.2. Explaratory Data Analysis.

4.3. Feature Engineering

4.4. Classification Modeling (Probability of Default).

4.5. Target variable creation for risk evaluation and assesment.

4.6. Regression Modeling.

4.7. Pipelines Creation (Classification and Regression).

4.8. UI: App Creation.

### 4.1 Data Preprocessing:
- The dataset contains **112** Columns and  **134529** Rows Range Index
- Removing the columns having missing value for then 40% missing values . after remove null columns now we have **77** columns and **134529** Rows for further use.
- Apart from missing value features there are some features which will have no role in default prediction like 'ReportAsOfEOD', 'LoanId', 'LoanNumber', 'ListedOnUTC', 'DateOfBirth' (**because age is already present**), 'BiddingStartedOn','UserName','NextPaymentNr','NrOfScheduledPayments','IncomeFromPrincipalEmployer', 'IncomeFromPension','IncomeFromFamilyAllowance', 'IncomeFromSocialWelfare','IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther' (**As Total income is already present which is total of all these income**), 'LoanApplicationStartedDate','ApplicationSignedHour', 'ApplicationSignedWeekday','ActiveScheduleFirstPaymentReached', 'PlannedInterestTillDate', 'LastPaymentOn', 'ExpectedLoss', 'LossGivenDefault', 'ExpectedReturn', 'ProbabilityOfDefault', 'PrincipalOverdueBySchedule', 'StageActiveSince', 'ModelVersion','WorseLateCategory'.
- Now we have **48** columns and **134529** Rows for further use.
- Here, status is the variable which help us in creating target variable. The reason for not making status as target variable is that it has three unique values **current, Late and repaid**. There is no default feature but there is a feature **default date** which tells us when the borrower has defaulted means on which date the borrower defaulted. So, we will be combining **Status** and **Default date** features for creating target  variable.The reason we cannot simply treat Late as default because it also has some records in which actual status is Late but the user has never defaulted i.e., default date is null. So we will first filter out all the current status records because they are not matured yet they are current loans.
- Now we have **48** columns and **77394** Rows for further use.
```
# Replace any Date by Defalut
for i in range(len(df.DefaultDate.values)):
    if df.DefaultDate.values[i] != 'NoDefault':
        df.DefaultDate.values[i] = 'Default'

# Rename DefaultDate column to LoanStatus
df.rename(columns={'DefaultDate':'LoanStatus'}, inplace = True)
 ```
 - We sew in numeric column distribution there are many columns which are present as numeric but they are actually categorical as per data description such as Verification Type, Language Code, Gender, Use of Loan, Education, Marital Status,EmployementStatus, OccupationArea etc.., So we will convert these features to categorical features.
 - Now, we have a clean optimized dataset, and we're ready for EDA.

### 4.2 Exploratory Data Analysis (EDA):
- While examining the data set through visualizations, There're some interesting trands showed in the data as the next few images suggest...

![pic_01](https://user-images.githubusercontent.com/93732090/209706346-3b03f75f-894f-4e2d-8e6c-b136ba34ef46.png)


- **Now setting the focus of our Exploration Analysis on the Defaulters only, we see that...**

**In the Categorical Attributes**

![pic_02](https://user-images.githubusercontent.com/93732090/209706675-3115a5c2-2f61-483c-b913-a4861166a00d.png)

![pic_03](https://user-images.githubusercontent.com/93732090/209706686-34f3ee2a-0889-4ffd-ae92-91d5dce9e4f5.png)

![pic_04](https://user-images.githubusercontent.com/93732090/209706691-e9d53626-29b4-4523-8cdf-9c422a18718d.png)

![pic_05](https://user-images.githubusercontent.com/93732090/209706702-c7bc3e86-379f-457e-8282-98d420d1bb82.png)

![pic_06](https://user-images.githubusercontent.com/93732090/209706715-b7c3c408-6ce6-4ff3-bff5-ffbddf4ef9dc.png)

![pic_07](https://user-images.githubusercontent.com/93732090/209706721-2b2daf27-7ae6-4d1f-9f88-1aebad5d7f67.png)


**In the Numerical Attributes**
```
fig, axs = plt.subplots(nrows= 3)

sns.histplot(df.Age[df.LoanStatus=='Default'], ax=axs[0]);
sns.distplot(df.Age[df.LoanStatus=='Default'], ax=axs[1])
sns.boxplot(df.Age[df.LoanStatus=='Default'], ax=axs[2]);
```

![pic_08](https://user-images.githubusercontent.com/93732090/209706741-60af511a-a51a-4d4e-ab30-4e9c5b039acd.png)

![pic_09](https://user-images.githubusercontent.com/93732090/209706756-f77c4878-5345-400e-99a8-3c8f5949f58a.png)


![pic_10](https://user-images.githubusercontent.com/93732090/209706813-fc55645d-c6b4-46b4-9ecf-9f9d62b82061.png)


- After we're throughly know evry attribute in the dataset, It's time for Feature Engineering...

### 4.3 Feature Engineering

a. Handling Missing Values
b. Handling Outliers
c. Feature Selection
d. Categorical Feature Encoding
e. Feature Scaling
f. Feature Extraction and Dimensionality reduction using PCA
g. Splitting Data into train and test sets.

**b. Handling outliers**
```
# Loop for replacing outliers above upper bound with the upper bound value:
for column in df.select_dtypes([float, int]).columns :
   
    col_IQR = df[column].quantile(.75) - df[column].quantile(.25)
    col_Max =  df[column].quantile(.75) + (1.5*col_IQR)
    df[column][df[column] > col_Max] =  col_Max
    
# Loop for replacing outliers under lower bound with the lower bound value:
for column in df.select_dtypes([float, int]).columns :
    col_IQR = df[column].quantile(.75) - df[column].quantile(.25)
    col_Min =  df[column].quantile(.25) - (1.5*col_IQR)
    df[column][df[column] < col_Min] =  col_Min
```
**c. Feature Selection**
- Define Highly Correlated attributes and handle them to avoid intercorrelation.
```
# A function to select highly correlated features.
def Correlation(dataset, threshold): 
    correltated_features = set() # as a container of highly correlated features
    correlation_matrix = dataset.corr()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                column_name = correlation_matrix.columns[i]
                correltated_features.add(column_name)
    return correltated_features
    
# let's selected features with a correlation factor > 0.8
Correlation(df, 0.8)

# Now we can drop these features from our dataset
df.drop(columns= ['Amount', 'AmountOfPreviousLoansBeforeLoan', 'NoOfPreviousLoansBeforeLoan'], inplace = True )
```

**d. Categorical feture Encoding**
```
Target_feature = df.LoanStatus
Ind_features   = df.drop(columns = ['LoanStatus'])

# Let's perform categorical features encoding:
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

# Target_feature Encoding:
Target_feature = LE.fit_transform(Target_feature)

# Ind_features Encoding:
for feature in Ind_features.select_dtypes([object, bool]).columns:
    Ind_features[feature]= LE.fit_transform(Ind_features[feature])
```

**e. Feature Scaling**
```
from sklearn.preprocessing import StandardScaler 
  Scalar = StandardScaler()
  Ind_features = Scalar.fit_transform(Ind_features)
```

**f. Feature Extraction and Dimensionality-reduction using (PCA)**
```
# importing PCA class
from sklearn.decomposition import PCA
# Create a PCA object with number of component = 25
pca2 = PCA(n_components = 25) 
# Let's fit our data using PCA
Ind_features_pca = pca2.fit_transform(Ind_features)
# Percentage of information we have after apllying 2-d PCA
sum(pca2.explained_variance_ratio_) * 100
```
- Using 2-d PCA we're preseved **94.76%** of information.

![pic_11](https://user-images.githubusercontent.com/93732090/209706945-a9f42648-d939-4c69-85f9-311380a2c48b.png)


**g. Splitting data into training and testing sets**
```
X = Ind_features_pca
y = Target_feature

# Let's use Train Test Split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,
                                                    train_size = .75, stratify=y)
```
- We have preserved 25% of the data to test after modeling.

### 4.4 Classification Modeling (Probability of Default)
- In this step we'll be training 4 different Models using default settings in scikit-leran, and with Hyperparameter tunning using RandomizedCV to select the highest performance model to intergrate later into the classification pipeline.
- We used classification_report(precision | recall | f1-score ), confusion_matrix, accuracy_score, and roc_auc_score metrics from sklearn.metrics to asses each model.
- The models used for classification are...
```
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# For hyperparameter Tunning
from sklearn.model_selection import RandomizedSearchCV
```
- The best performing Classification Model at this stage of Analysis was GradientBoostingClassifier, with the follwing results on evaluating metrics

![pic_12](https://user-images.githubusercontent.com/93732090/209706964-de0c8915-874b-4187-a694-5172038b8188.png)


### 4.5 Target variable creation for risk evaluation and assesment
After a thorught reaserch to identify 3 new Procedures to evaluate the 3 target assesment features agrred upon on the Planning stage,
We came up with theses Algorithms...

**Eligible Loan Amount**
-- add pic from Report -- 

**Equaty Monthly Installments**
-- add pic from Report --

**Preferred ROI**
-- add pic from Report --

- The loan tenure that has been used in the 3 procedures above is calculated as follows...

**Loan Tenure**

-- add pic from Report --

- Theses 3 Target variable Creation Steps were added at the preprocessing Stage of the Development Cycle along with LoanStatus variable creation.
- Know we;re ready to go a head with Mu;tiRegression Modeling.

### 4.6 Regression Modeling
- In order to prepare the dataset for MultiRegeression Modeling we had to revisit the dataset at the Stage of Features Engineering,
- There we changed 2 major steps of FE...

      1. Feature Selection
      2. Categorical Fetaure Encoding      
      3. PCA
      
**1. Feature Selection**

- The highly correlated set of attributes needs to be dropped to avoid intercorrelation in the dataset has changed to be...

```
# Now we can drop these features from our dataset
X.drop(columns= ['LoanTenure', 'ROI', 'Amount', 'TotalAmount', 'Total_Loan_Amnt', 'AmountOfPreviousLoansBeforeLoan', 'NoOfPreviousLoansBeforeLoan'], inplace = True )
```


**2. Categorical Fetaure Encoding**

- Following a dummy variables creating approach instead of label encoding.

```
X = pd.get_dummies(X, drop_first=True)
```

**3. PCA**

- Now the data set has **143** attributes.

- Changing the PCA usage (n_components=110) to go along with the new Changes in the dataset...

```
PCA(n_components=110)
```
Which allowed preseving almsot 99% of information of the dataset.

**Classification Modeling Changes**

- These Changes in Feture Enginnering in turn afftected our Choice of the Best performing Classification Model, which now is 
```
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=0)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
```

![pic_13](https://user-images.githubusercontent.com/93732090/209707011-65c83441-8189-440b-b958-bbd8f9965ba6.png)


**MultiRegression Modeling**

- We have trained 3 different Models with their default values, and with Hyperparameter Tunning to select the higest performing Model.
- We have selected mean_square_error, mean_square_error_percentage, and r2_score from sklearn.metrics to evaluate selected models.
- The three choosen models (all linear) are...
```
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# for hyperparameter tunning
from sklearn.model_selection import RandomizedSearchCV
```

- The best performing Regreesion Model was Ridge with default parameters by scikit-learn.
```
from sklearn.linear_model import Ridge

rid_reg = Ridge()

rid_reg.fit(X_train, y_train)

y_pred_base = rid_reg.predict(X_test)
```

![pic_14](https://user-images.githubusercontent.com/93732090/209707032-79c80f0d-a195-4caa-9884-7065847401af.png)


### 4.7 Pipelines Creation (Classification and Regression)

- Now Automating all the Important steps of Feature Engieering, Modeling into 2 Pipelines.
- There'll be 2 common steps of the Pipelines which are
           - Features Scaling.
           - PCA
- for classification Pipeline, we'll use Logostic Regression
- for Regression Pipeline, we'll use Ridge.
- The output is 2 pickle files for the 2 pipelines to be used in UI.

```
# Create Classification Pipeline
pipeline_class = Pipeline([
    ('stdscaler', StandardScaler()),
    ('pca', PCA(n_components=110)),
    ('Classifier', LogisticRegression(random_state=0))
])

# Create Regression Model
pipeline_reg = Pipeline([
    ('stdscaler', StandardScaler()),
    ('pca', PCA(n_components=110)),
    ('Regressor', Ridge(random_state=0))
])
```

- Dumping 2 pipeline files
```
pickle.dump(pipeline_class, open('pipeline_class.pkl', 'wb'))
pickle.dump(pipeline_reg, open('pipeline_reg.pkl', 'wb'))
```

### 4.8. UI: App Creation
- Using Flask API, and simple HTML5, and CSS; we created a web application with v01 as specified during the planning Step of the analysis.
- v02 is currently under development
- During v01, developed files contained:
      1. app.py --- for Flask to run the app and steer it's way around different files.
      
      2. pipelines.py --- fpr preprocessing of input data from the Client and make it match the attributes expected by the Pipelines file, furthermore run the pipelines files.
      
      3. index.html --- v01 fourm (individual borrower entry using a fourm)
      
      4. submit.html --- v01 output fourm of expected assesment criteria's
      
  
## 5. Deployment
- After agrreing on AWS-EC2 as the deployment platform for this app, we deployed the app for production.
- [App link](http://13.126.68.187:8080/)

## 6. Conclusion:
- We've fullfilled all assemnet reauiremnets of the Client.
- We've developed 1st vesion of UI requiremnet (v01) which is Single borrower data entry uding a fourm.
- (v02) Multiple borrower data entry via CSV file upload is under development...
