# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:57:39 2020

@author: allwin
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

data_loan=pd.read_csv('D:\project\Python Project - Bank Lending\XYZCorp_LendingData.txt',encoding='utf-8',sep='\t')

data_loan.dtypes.value_counts()
data_loan.describe()
aa=data_loan.isnull().sum()

# Drop irrelevant columns
data_loan.drop(['id', 'member_id', 'emp_title', 'desc', 'zip_code', 'title'], axis=1, inplace=True)

####trying to plot the target variable
data_loan['default_ind'].value_counts().plot.bar()

# Lets' transform the issue dates by year.
####utilising the year from issue_d
dt_series = pd.to_datetime(data_loan['issue_d'])
data_loan['year'] = dt_series.dt.year
data_loan['year'] = data_loan['year'].astype(object)


plt.figure(figsize= (8,5))
plt.ylabel('loan_amount issued')
plt.xlabel('year')
sns.barplot('year', 'loan_amnt', data=data_loan, palette='tab10')

####it seems that loan issued are more from 2012-2015

plt.figure(figsize= (8,5))
plt.ylabel('default')
plt.xlabel('year')
sns.barplot('year', 'default_ind', data=data_loan,palette='tab10')

#####EDA
fig, ax = plt.subplots(1, 3, figsize=(12,5))
sns.distplot(data_loan['loan_amnt'], ax=ax[0])
sns.distplot(data_loan['funded_amnt'], ax=ax[1])
sns.distplot(data_loan['funded_amnt_inv'], ax=ax[2])
ax[1].set_title("Amount Funded by the Lender")
ax[0].set_title("Loan Applied by the Borrower")
ax[2].set_title("Total committed by Investors")

#### Loan Purpose may also give us an insight
data_loan.purpose.value_counts(ascending=False).plot.bar(figsize=(10,5))
plt.xlabel('purpose'); plt.ylabel('Density'); plt.title('Purpose of loan');
####debt Consolidation is also an another factor for which loan is taken

## Loan issued by regions
# Make a list with each of the regions by state.

west = ['WA','CA', 'OR', 'UT','ID','CO', 'NV', 'NM', 'AK', 'MT', 'HI', 'WY']
south_east = ['AZ', 'TX', 'OK','GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN' ]
mid_west = ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND']
north_east = ['CT', 'NY', 'PA', 'NJ', 'RI','MA', 'MD', 'VT', 'NH', 'ME']

data_loan['region'] = np.nan

def fix_regions(addr_state):
        if addr_state in west:
            return 'west'
        elif addr_state in south_east:
            return 'south east'
        elif addr_state in mid_west:
            return 'mid west'
        else:
            return 'north east'
        
data_loan['region'] = data_loan['addr_state'].apply(fix_regions)

date_amt_region = data_loan[['loan_amnt','issue_d','region']]
plt.style.use('seaborn')
cmap = plt.cm.Paired
by_issued_amount = date_amt_region.groupby(['issue_d', 'region']).loan_amnt.sum()
by_issued_amount.unstack().plot(stacked=False, colormap=cmap, grid=False, legend=True, figsize=(15,6))

plt.title('Loans issued by Region', fontsize=16)

data_loan.drop(['addr_state'],1,inplace=True)


## Converting emp length into integer

data_loan.emp_length.unique()
data_loan.loc[data_loan['emp_length']=='10+ years','emp_len'] = 10
data_loan.loc[data_loan['emp_length']=='<1 year','emp_len'] = .5
data_loan.loc[data_loan['emp_length']=='1 year','emp_len'] = 1
data_loan.loc[data_loan['emp_length']=='3 years','emp_len'] = 3
data_loan.loc[data_loan['emp_length']=='8 years','emp_len'] = 8
data_loan.loc[data_loan['emp_length']=='9 years','emp_len'] = 9
data_loan.loc[data_loan['emp_length']=='4 years','emp_len'] = 4
data_loan.loc[data_loan['emp_length']=='5 years','emp_len'] = 5
data_loan.loc[data_loan['emp_length']=='6 years','emp_len'] = 6
data_loan.loc[data_loan['emp_length']=='2 years','emp_len'] = 2
data_loan.loc[data_loan['emp_length']=='7 years','emp_len'] = 7
data_loan.emp_len.fillna(value=0,inplace=True)
data_loan.emp_len.unique()

data_loan['emp_len'] = data_loan['emp_len'].astype(int)
data_loan.drop(['emp_length'],1,inplace=True)


# Loan issued by Region ,Credit Score and grade
plt.style.use('seaborn-ticks')


f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(16,12))

regional_interest_rate = data_loan.groupby(['year', 'region']).int_rate.mean()
regional_interest_rate.unstack().plot(kind='line',  stacked=True,colormap=cmap, grid=False,
                                      legend=False, figsize=(16,12),ax=ax1)

regional_emp_length = data_loan.groupby(['year', 'region']).emp_len.mean()
regional_emp_length.unstack().plot(kind='line',  stacked=True,colormap=cmap, grid=False,
                                      legend=False, figsize=(16,12),ax=ax2)

regional_dti = data_loan.groupby(['year', 'region']).dti.mean()
regional_dti.unstack().plot(kind='line',  stacked=True,colormap=cmap, grid=False,
                                      legend=False, figsize=(16,12),ax=ax3)

regional_interest_rate = data_loan.groupby(['year', 'region']).annual_inc.mean()
regional_interest_rate.unstack().plot(kind='line',  stacked=True,colormap=cmap, grid=False,
                                      legend=False, figsize=(16,12),ax=ax4)
ax1.set_title('average interest rate vs region'),ax2.set_title('average emp_length by region')
ax3.set_title('average dti by region'),ax4.set_title('average annual income by region')

ax4.legend(bbox_to_anchor=(-1.0, -0.5, 1.8, 0.1), loc=10,prop={'size':12},
           ncol=5, mode="expand", borderaxespad=0.)


#annual_inc is a float value i.e,numerical dividing that by three segments and dummifying that
data_loan['income_category'] = np.nan
data_loan.loc[data_loan['annual_inc'] <= 100000,'income_category'] = 'Low'
data_loan.loc[(data_loan['annual_inc'] > 100000) & (data_loan['annual_inc'] <= 200000),'income_category'] = 'Medium'
data_loan.loc[data_loan['annual_inc'] > 200000,'income_category'] = 'High'

fig, ((ax1, ax2), (ax3, ax4))= plt.subplots(nrows=2, ncols=2, figsize=(14,8))
plt.style.use('bmh')
sns.barplot(x="income_category", y="loan_amnt", data=data_loan, ax=ax1 )
sns.barplot(x="income_category", y="default_ind", data=data_loan, ax=ax2)
sns.barplot(x="income_category", y="emp_len", data=data_loan, ax=ax3)
sns.barplot(x="income_category", y="int_rate", data=data_loan, ax=ax4)
plt.tight_layout(h_pad=1.5)


## Bad Loans
defaulter = data_loan.loc[data_loan['default_ind']==1]
plt.figure(figsize=(16,16))
plt.subplot(211)
sns.barplot(data=defaulter,x = 'home_ownership',y='loan_amnt',hue='default_ind')
plt.subplot(212)
sns.barplot(data=defaulter,x='year',y='loan_amnt',hue='home_ownership')

#####no proper payment plan
sns.countplot(data_loan['pymnt_plan'],hue=data_loan['default_ind'])

#####irresopective to any grade default is there
sns.countplot(data_loan['grade'],hue=data_loan['default_ind'])

#####there are six columns where missing values are very less so we can treat them
'last_credit_pull_d,,collections_12_mths_ex_med,,revol_util'
'tot_coll_amt,tot_cur_bal,total_rev_hi_lim'

fdfd=data_loan.isnull().sum()
data_loan.collections_12_mths_ex_med.unique()
data_loan['collections_12_mths_ex_med'].value_counts().plot.bar()
data_loan['collections_12_mths_ex_med'].value_counts()
#####Number of collections in 12 months excluding medical collections
#####large number are zeros why no collection????
data_loan.collections_12_mths_ex_med.fillna(value=0,inplace=True)

data_loan['revol_util'].value_counts()
data_loan['revol_util'].describe()
data_loan['revol_util'].fillna((data_loan['revol_util'].mean()), inplace=True)

data_loan.last_credit_pull_d=data_loan.last_credit_pull_d.fillna(data_loan['last_credit_pull_d'].value_counts().idxmax())

data_loan['tot_coll_amt'].fillna((data_loan['tot_coll_amt'].mean()), inplace=True)

data_loan['tot_cur_bal'].fillna((data_loan['tot_cur_bal'].mean()), inplace=True)

data_loan['total_rev_hi_lim'].fillna((data_loan['total_rev_hi_lim'].mean()), inplace=True)

########only six possible columns could be treated#######
data_loan['earliest_cr_line']= data_loan['earliest_cr_line'].apply(lambda s:int(s[-4:]))
data_loan['last_credit_pull_d']= data_loan['last_credit_pull_d'].apply(lambda s:int(s[-4:]))
data_loan['credit_age']= data_loan['last_credit_pull_d'] - data_loan['earliest_cr_line']
data_loan.drop(['earliest_cr_line','last_credit_pull_d'],1,inplace=True)
data_loan.credit_age.unique()
data_loan['credit_age'] = data_loan['credit_age'].astype(float)

########deleting columns with large number of missing values treating them would lead to biased modelling

data_loan = data_loan.dropna(axis=1)

#######data_loan##################################


#according to data types creating dataframes
data_ord=data_loan[['emp_len','default_ind']]
data_nume=data_loan.select_dtypes(include='float64')
data_nom=data_loan.select_dtypes(include='object')

####################### data_nume #################################

'these numerical cannot be quantile cut for highly discontinuous data distributions'

'Create correlation matrix'
data_nume_scaled=data_nume.copy()

#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = data_nume_scaled.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

corr_matrix = data_nume_scaled.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
data_nume_scaled=data_nume_scaled.drop(data_nume_scaled[to_drop], axis=1)

# scaling
scaler = preprocessing.MinMaxScaler().fit(data_nume_scaled)
scaled_numeric = scaler.transform(data_nume_scaled)
data_nume_scaled=pd.DataFrame(scaled_numeric, columns=data_nume_scaled.columns.tolist())

####################### data_nom #################################
#the nominal datas are dummified ,except the column issue_d because that will be used for train and test split
non_dummy_cols = ['issue_d'] 
# Takes all 47 other columns
dummy_cols = list(set(data_nom.columns) - set(non_dummy_cols))
data_nom_dum = pd.get_dummies(data_nom, columns=dummy_cols)

uniqueValues = data_nom.nunique()

######################data_ord##########################
data_ord.year.unique()
data_ord.emp_len.unique()

'since emp_len seems to look like label encoded data'
"so we rae trying to keep the data without any change"

#Log reg , KNN, SVM will use Scaled numeric, labbel & dummiees for nom will both be used
'lets concat the dataframes'

data_prepared = pd.concat([data_nom_dum,data_nume_scaled,data_ord], axis = 1)

'for splitting purpose'

data_prepared['issue_d'] = pd.to_datetime(data_prepared['issue_d'])


# Creating train and test data set
# According to problem statement given
'The data should be divided into train ( June 2007 - May 2015 )'
'and out-of-time test ( June 2015 - Dec 2015 ) data.'

train_data = data_prepared[data_prepared['issue_d'] < '2015-6-01']
test_data = data_prepared[data_prepared['issue_d'] >= '2015-6-01']


####################splitting x_test,y_test,x_train,y_train
y_test=test_data['default_ind'].copy()

train_data=train_data.drop(['issue_d'],axis=1)
test_data=test_data.drop(['issue_d','default_ind'],axis=1)

X_test=test_data.copy()

##############################################################
d_DT_WS=train_data.copy()

x_ws = d_DT_WS.iloc[:,0:-1]
y_ws = d_DT_WS['default_ind']

'since machine cant read a big data i.e, memory error we are splitting '
'the train data by train and validation so that it becomes easy for the machine to get trained'

X_train, X_val, y_train, y_val = train_test_split(x_ws, y_ws, test_size = 0.3,shuffle=True, random_state = None)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

sm = SMOTE(k_neighbors=10, random_state=44)
X_res_train, y_res_train = sm.fit_sample(X_train, y_train)


logreg = LogisticRegression()
logreg.fit(X_res_train, y_res_train)

y_pred = logreg.predict(X_val)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_val, y_val)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_val, y_pred)
print(confusion_matrix)
print("Accuracy:",metrics.accuracy_score(y_val, y_pred))
print("Precision:",metrics.precision_score(y_val, y_pred))
print("Recall:",metrics.recall_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

logit_roc_auc = roc_auc_score(y_val, logreg.predict(X_val))
fpr, tpr, thresholds = roc_curve(y_val, logreg.predict_proba(X_val)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

#####predicting our test data 

y_prediction = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_prediction)
print(confusion_matrix)
print("Accuracy:",metrics.accuracy_score(y_test, y_prediction))
print("Precision:",metrics.precision_score(y_test, y_prediction))
print("Recall:",metrics.recall_score(y_test, y_prediction))
print(classification_report(y_test, y_prediction))

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

#####################################################






