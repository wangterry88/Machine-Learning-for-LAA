#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
#import selenium
os.chdir("../../../LAA/")


# In[2]:


# !pip uninstall scikit-learn -y --user
# !pip install scikit-learn==1.0.2 --user
# !pip install lightgbm --user
# !pip install eli5 --user
# !pip install scikit-optimize --user
# !pip install xgboost==1.2.0 --user
# !pip install tableone --user
# !pip install openpyxl --user
# !pip install -U yellowbrick --user
# !pip install matplotlib-venn --user
#!pip install venn
# !pip install --upgrade --force-reinstall venn
# !conda install --yes -c rapidsai -c nvidia -c conda-forge  cuml=22.04 cudf=22.04 --user
# !conda install --yes -c conda-forge screen --user


# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import precision_score, roc_curve, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
from tableone import TableOne
import pandas as pd
pd.options.mode.chained_assignment = None
import scipy
from scipy import stats
# import cuml
# import cupy as cp


# In[4]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# In[5]:


# !python -m pip show scikit-learn
# !python -m pip freeze


# # Load data (Imputed)

# In[6]:


laa=pd.read_csv("./new-data-220328/LAA-clinic.miRNA.0328.imputed",sep="\t", engine='python')


# In[7]:


#print("Dataset has {} entries and {} features".format(*laa.shape))


# In[8]:


laa.head()


# In[9]:


laa=laa.drop(columns=['ID'])


# # Define model columns

# In[10]:


columns_clinical = [ 'Group','age','sex','HT','DM','Smoking','Alcohol','FHx stroke','CKD','Lipid_drug','DM_drug', 'HTN_drug',
                    'BH', 'BW', 'BMI','waistline', 'Hip','SBP', 'DBP','MBP','HeartRate', 'Homocysteine', 'AcSugar', 'HsCRP', 
                    'HDL', 'LDL', 'TG', 'CHOL', 'UA', 'Creatinine']


# In[11]:


len(columns_clinical)


# In[12]:


columns_metabo = [ 'Group','C0','C10','C101','C102','C12','C12DC','C121','C14','C141','C141OH','C142','C142OH','C16', 'C16OH', 
                  'C161', 'C161OH', 'C162', 'C162OH', 'C18', 'C181', 'C181OH', 'C182', 'C2', 'C3','C3DCC4OH', 'C3OH', 'C31', 'C4', 
                  'C41', 'C6C41DC', 'C5', 'C5MDC', 'C5OHC3DCM', 'C51', 'C51DC', 'C5DCC6OH', 'C61', 'C7DC', 'C8', 'C9', 'Ala', 'Arg', 
                  'Asn', 'Asp', 'Cit', 'Gln', 'Glu','Gly', 'His', 'Ile', 'Leu', 'Lys', 'Met', 'Orn', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 
                  'Tyr', 'Val', 'ADMA', 'alphaAAA', 'Creatinine_MS', 'Kynurenine', 'Sarcosine','t4OHPro', 'Taurine', 'SDMA', 'lysoPCaC160',
                  'lysoPCaC161', 'lysoPCaC170', 'lysoPCaC180', 'lysoPCaC181', 'lysoPCaC182', 'lysoPCaC203', 'lysoPCaC204', 'lysoPCaC240', 
                  'lysoPCaC260', 'lysoPCaC261', 'lysoPCaC281', 'PCaaC240', 'PCaaC281', 'PCaaC300','PCaaC320', 'PCaaC321', 'PCaaC323', 
                  'PCaaC341','PCaaC342', 'PCaaC343','PCaaC344', 'PCaaC360', 'PCaaC361', 'PCaaC362', 'PCaaC363', 'PCaaC364', 'PCaaC365', 
                  'PCaaC366', 'PCaaC380', 'PCaaC383', 'PCaaC384', 'PCaaC385', 'PCaaC386', 'PCaaC402', 'PCaaC403', 'PCaaC404', 'PCaaC405',
                  'PCaaC406', 'PCaaC420', 'PCaaC421', 'PCaaC422', 'PCaaC424', 'PCaaC425', 'PCaaC426', 'PCaeC300', 'PCaeC302', 'PCaeC321',
                  'PCaeC322', 'PCaeC340', 'PCaeC341', 'PCaeC342', 'PCaeC343','PCaeC360', 'PCaeC361', 'PCaeC362','PCaeC363', 'PCaeC364',
                  'PCaeC365', 'PCaeC380', 'PCaeC382', 'PCaeC383', 'PCaeC384', 'PCaeC385','PCaeC386', 'PCaeC401', 'PCaeC402','PCaeC403',
                  'PCaeC404', 'PCaeC405', 'PCaeC406', 'PCaeC420', 'PCaeC421', 'PCaeC422', 'PCaeC423', 'PCaeC424', 'PCaeC425', 'PCaeC443',
                  'PCaeC444', 'PCaeC445', 'PCaeC446', 'SMOHC141', 'SMOHC161','SMOHC221','SMOHC222','SMOHC241','SMC160','SMC161','SMC180',
                  'SMC181','SMC202','SMC240','SMC241','SMC260','SMC261']


# In[13]:


len(columns_metabo)


# In[14]:


columns_all=['Group','age','sex','HT','DM','Smoking','Alcohol','FHx stroke','CKD','BH', 'BW', 'BMI','waistline', 'Hip',
           'SBP', 'DBP','MBP','HeartRate', 'Homocysteine', 'AcSugar', 'HsCRP', 'HDL', 'LDL', 'TG', 'CHOL', 'UA', 
           'Creatinine','Lipid_drug','DM_drug', 'HTN_drug',
            'C0','C10','C101','C102','C12','C12DC','C121','C14','C141','C141OH','C142','C142OH','C16', 'C16OH', 'C161', 
             'C161OH', 'C162', 'C162OH', 'C18', 'C181', 'C181OH', 'C182', 'C2', 'C3','C3DCC4OH', 'C3OH', 'C31', 'C4', 'C41', 
             'C6C41DC', 'C5', 'C5MDC', 'C5OHC3DCM', 'C51', 'C51DC', 'C5DCC6OH', 'C61', 'C7DC', 'C8', 'C9', 'Ala', 'Arg', 'Asn', 
             'Asp', 'Cit', 'Gln', 'Glu','Gly', 'His', 'Ile', 'Leu', 'Lys', 'Met', 'Orn', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 
             'Val', 'ADMA', 'alphaAAA', 'Creatinine_MS', 'Kynurenine', 'Sarcosine','t4OHPro', 'Taurine', 'SDMA', 'lysoPCaC160',
             'lysoPCaC161', 'lysoPCaC170', 'lysoPCaC180', 'lysoPCaC181', 'lysoPCaC182', 'lysoPCaC203', 'lysoPCaC204', 'lysoPCaC240',
             'lysoPCaC260', 'lysoPCaC261', 'lysoPCaC281', 'PCaaC240', 'PCaaC281', 'PCaaC300','PCaaC320', 'PCaaC321', 'PCaaC323',
             'PCaaC341','PCaaC342', 'PCaaC343','PCaaC344', 'PCaaC360', 'PCaaC361', 'PCaaC362', 'PCaaC363', 'PCaaC364', 'PCaaC365',
             'PCaaC366', 'PCaaC380', 'PCaaC383', 'PCaaC384', 'PCaaC385', 'PCaaC386', 'PCaaC402', 'PCaaC403', 'PCaaC404', 'PCaaC405',
             'PCaaC406', 'PCaaC420', 'PCaaC421', 'PCaaC422', 'PCaaC424', 'PCaaC425', 'PCaaC426', 'PCaeC300', 'PCaeC302', 'PCaeC321',
             'PCaeC322', 'PCaeC340', 'PCaeC341', 'PCaeC342', 'PCaeC343','PCaeC360', 'PCaeC361', 'PCaeC362','PCaeC363', 'PCaeC364',
             'PCaeC365', 'PCaeC380', 'PCaeC382', 'PCaeC383', 'PCaeC384', 'PCaeC385','PCaeC386', 'PCaeC401', 'PCaeC402','PCaeC403',
             'PCaeC404', 'PCaeC405', 'PCaeC406', 'PCaeC420', 'PCaeC421', 'PCaeC422', 'PCaeC423', 'PCaeC424', 'PCaeC425', 'PCaeC443',
             'PCaeC444', 'PCaeC445', 'PCaeC446', 'SMOHC141', 'SMOHC161','SMOHC221','SMOHC222','SMOHC241','SMC160','SMC161','SMC180',
             'SMC181','SMC202','SMC240','SMC241','SMC260','SMC261']


# In[15]:


len(columns_all)


# # Define model (3 kind of models)

# In[16]:


laa_clinical=laa[columns_clinical]
laa_clinical.head()


# In[17]:


laa_metabo=laa[columns_metabo]
laa_metabo.head()


# In[18]:


laa_all=laa[columns_all]
laa_all.head()


# # Get Ready data (After model feature selection)

# ## Select 3 kind of Model

# In[98]:


# laa_clean=laa_clinical
# laa_clean=laa_metabo
laa_clean=laa_all


# In[99]:


#Recode factor
laa_clean['Group'] = np.where(laa_clean['Group']== "LAA", 1, 0)
laa_clean.columns


# In[100]:


# All model
x = laa_clean.loc[:,'age':'SMC261']
y = laa_clean.loc[:,'Group']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state =2018, shuffle = True)


# In[101]:


X_train.head()


# # Standard Scalar

# In[102]:


# (Clinical) features need to be standard scalar

std_sc_clinical = ['age','BH', 'BW', 'BMI','waistline', 'Hip', 'SBP', 
                 'DBP', 'MBP', 'HeartRate', 'Homocysteine','AcSugar', 
                 'HsCRP', 'HDL', 'LDL', 'TG', 'CHOL', 'UA', 'Creatinine']



# (Metabolite)  features need to be standard scalar

std_sc_metabo = [ 'C0','C10','C101','C102','C12','C12DC','C121','C14','C141','C141OH','C142','C142OH','C16', 'C16OH', 
                  'C161', 'C161OH', 'C162', 'C162OH', 'C18', 'C181', 'C181OH', 'C182', 'C2', 'C3','C3DCC4OH', 'C3OH', 'C31', 'C4', 
                  'C41', 'C6C41DC', 'C5', 'C5MDC', 'C5OHC3DCM', 'C51', 'C51DC', 'C5DCC6OH', 'C61', 'C7DC', 'C8', 'C9', 'Ala', 'Arg', 
                  'Asn', 'Asp', 'Cit', 'Gln', 'Glu','Gly', 'His', 'Ile', 'Leu', 'Lys', 'Met', 'Orn', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 
                  'Tyr', 'Val', 'ADMA', 'alphaAAA', 'Creatinine_MS', 'Kynurenine', 'Sarcosine','t4OHPro', 'Taurine', 'SDMA', 'lysoPCaC160',
                  'lysoPCaC161', 'lysoPCaC170', 'lysoPCaC180', 'lysoPCaC181', 'lysoPCaC182', 'lysoPCaC203', 'lysoPCaC204', 'lysoPCaC240', 
                  'lysoPCaC260', 'lysoPCaC261', 'lysoPCaC281', 'PCaaC240', 'PCaaC281', 'PCaaC300','PCaaC320', 'PCaaC321', 'PCaaC323', 
                  'PCaaC341','PCaaC342', 'PCaaC343','PCaaC344', 'PCaaC360', 'PCaaC361', 'PCaaC362', 'PCaaC363', 'PCaaC364', 'PCaaC365', 
                  'PCaaC366', 'PCaaC380', 'PCaaC383', 'PCaaC384', 'PCaaC385', 'PCaaC386', 'PCaaC402', 'PCaaC403', 'PCaaC404', 'PCaaC405',
                  'PCaaC406', 'PCaaC420', 'PCaaC421', 'PCaaC422', 'PCaaC424', 'PCaaC425', 'PCaaC426', 'PCaeC300', 'PCaeC302', 'PCaeC321',
                  'PCaeC322', 'PCaeC340', 'PCaeC341', 'PCaeC342', 'PCaeC343','PCaeC360', 'PCaeC361', 'PCaeC362','PCaeC363', 'PCaeC364',
                  'PCaeC365', 'PCaeC380', 'PCaeC382', 'PCaeC383', 'PCaeC384', 'PCaeC385','PCaeC386', 'PCaeC401', 'PCaeC402','PCaeC403',
                  'PCaeC404', 'PCaeC405', 'PCaeC406', 'PCaeC420', 'PCaeC421', 'PCaeC422', 'PCaeC423', 'PCaeC424', 'PCaeC425', 'PCaeC443',
                  'PCaeC444', 'PCaeC445', 'PCaeC446', 'SMOHC141', 'SMOHC161','SMOHC221','SMOHC222','SMOHC241','SMC160','SMC161','SMC180',
                  'SMC181','SMC202','SMC240','SMC241','SMC260','SMC261']



# (All) features need to be standard scalar

std_sc_all = ['age','BH', 'BW', 'BMI','waistline', 'Hip','SBP', 'DBP',
             'MBP','HeartRate', 'Homocysteine', 'AcSugar', 'HsCRP', 
             'HDL', 'LDL', 'TG', 'CHOL', 'UA', 'Creatinine',
             'C0','C10','C101','C102','C12','C12DC','C121','C14','C141','C141OH','C142','C142OH','C16', 
             'C16OH', 'C161', 'C161OH', 'C162', 'C162OH', 'C18', 'C181', 'C181OH', 'C182', 'C2', 'C3','C3DCC4OH',
             'C3OH', 'C31', 'C4', 'C41', 'C6C41DC', 'C5', 'C5MDC', 'C5OHC3DCM', 'C51', 'C51DC', 'C5DCC6OH', 'C61',
             'C7DC', 'C8', 'C9', 'Ala', 'Arg', 'Asn', 'Asp', 'Cit', 'Gln', 'Glu','Gly', 'His', 'Ile', 'Leu', 'Lys',
             'Met', 'Orn', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val', 'ADMA', 'alphaAAA', 'Creatinine_MS',
             'Kynurenine', 'Sarcosine','t4OHPro', 'Taurine', 'SDMA', 'lysoPCaC160', 'lysoPCaC161', 'lysoPCaC170', 
             'lysoPCaC180', 'lysoPCaC181', 'lysoPCaC182', 'lysoPCaC203', 'lysoPCaC204', 'lysoPCaC240', 'lysoPCaC260', 
             'lysoPCaC261', 'lysoPCaC281', 'PCaaC240', 'PCaaC281', 'PCaaC300','PCaaC320', 'PCaaC321', 'PCaaC323',
             'PCaaC341','PCaaC342', 'PCaaC343','PCaaC344', 'PCaaC360', 'PCaaC361', 'PCaaC362', 'PCaaC363', 'PCaaC364',
             'PCaaC365', 'PCaaC366', 'PCaaC380', 'PCaaC383', 'PCaaC384', 'PCaaC385', 'PCaaC386', 'PCaaC402', 'PCaaC403',
             'PCaaC404', 'PCaaC405', 'PCaaC406', 'PCaaC420', 'PCaaC421', 'PCaaC422', 'PCaaC424', 'PCaaC425', 'PCaaC426',
             'PCaeC300', 'PCaeC302', 'PCaeC321', 'PCaeC322', 'PCaeC340', 'PCaeC341', 'PCaeC342', 'PCaeC343','PCaeC360',
             'PCaeC361', 'PCaeC362','PCaeC363', 'PCaeC364', 'PCaeC365', 'PCaeC380', 'PCaeC382', 'PCaeC383', 'PCaeC384',
             'PCaeC385','PCaeC386', 'PCaeC401', 'PCaeC402','PCaeC403', 'PCaeC404', 'PCaeC405', 'PCaeC406', 'PCaeC420',
             'PCaeC421', 'PCaeC422', 'PCaeC423', 'PCaeC424', 'PCaeC425', 'PCaeC443', 'PCaeC444', 'PCaeC445', 'PCaeC446',
             'SMOHC141', 'SMOHC161','SMOHC221','SMOHC222','SMOHC241','SMC160','SMC161','SMC180','SMC181','SMC202','SMC240',
             'SMC241','SMC260','SMC261']


# ## Select the model features 

# In[103]:


# col_names=std_sc_clinical
# col_names=std_sc_metabo
col_names=std_sc_all


# In[104]:


scalar=StandardScaler()


# In[105]:


# Train data Standard scaler
features_train = X_train[col_names]
scaler_train = scalar.fit_transform(features_train)
X_train[col_names] = scaler_train


# In[106]:


# Test data Standard scaler
features_test = X_test[col_names]
scaler_test = scalar.transform(features_test)
X_test[col_names] = scaler_test


# In[107]:


X_train.head()


# In[108]:


X_test.head()


# In[109]:


print(len(X_train),len(X_test))


# # Receiver Operating Characteristic (ROC) with cross validation

# In[110]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold


# In[111]:


from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold


# In[112]:


X_cv = X_train
y_cv = y_train


# In[113]:


X_cv_nmp=X_cv.to_numpy()
y_cv_nmp=y_cv.to_numpy()


# # Logistic Regression

# In[114]:


# LogisticRegression
# Classification and ROC analysis
# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=10)
#classifier = svm.SVC(kernel="linear", probability=True)
classifier_LR = LogisticRegression(max_iter=3000)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=(10, 6))
for i, (train, test) in enumerate(cv.split(X_cv_nmp, y_cv_nmp)):
    classifier_LR.fit(X_cv_nmp[train], y_cv_nmp[train])
    viz = RocCurveDisplay.from_estimator(
        classifier_LR,
        X_cv_nmp[test],
        y_cv_nmp[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic of Logistic Regression",
)
ax.set_title('Receiver operating characteristic of Logistic Regression',fontsize= 18)
ax.legend(loc="lower right")
plt.show()


# In[115]:


len(X_cv_nmp),len(y_cv_nmp)


# In[116]:


from sklearn.model_selection import cross_validate

# train with 10 fold corss validation
#classifier_LR_cv = LogisticRegression(max_iter=3000)
classifier_LR = LogisticRegression(max_iter=3000)
scoring=['accuracy','roc_auc','recall','precision','f1']
scores_LR_cv = cross_validate(classifier_LR,X_cv_nmp,y_cv_nmp,cv=10,scoring=scoring)

#print(sorted(scores_LR_cv.keys()))
#print(scores_LR_cv['test_recall_macro'])
print("10 fold Accuracy: %0.2f (± %0.2f)" % (scores_LR_cv['test_accuracy'].mean(),scores_LR_cv['test_accuracy'].std()*2))
print("10 fold AUC: %0.2f (± %0.2f)" % (scores_LR_cv['test_roc_auc'].mean(),scores_LR_cv['test_roc_auc'].std()*2))
print("10 fold Recall: %0.2f (± %0.2f)" % (scores_LR_cv['test_recall'].mean(),scores_LR_cv['test_recall'].std()*2))
print("10 fold Precision: %0.2f (± %0.2f)" % (scores_LR_cv['test_precision'].mean(),scores_LR_cv['test_precision'].std()*2))
print("10 fold f1: %0.2f (± %0.2f)" % (scores_LR_cv['test_f1'].mean(),scores_LR_cv['test_f1'].std()*2))


# In[117]:


## Trained with 10-fold validation

classifier_LR.fit(X_train, y_train)

## test
predicted_prob_LR = classifier_LR.predict_proba(X_test)[:,1]
predicted = classifier_LR.predict(X_test)

#############################################################

## Accuray e AUC
accuracy = metrics.accuracy_score(y_test, predicted)
auc_test = metrics.roc_auc_score(y_test, predicted_prob_LR)
print("Accuracy (overall correct predictions):",  round(accuracy,2))
print("Auc:", round(auc_test,2))
    
## Precision e Recall
recall = metrics.recall_score(y_test, predicted)
precision = metrics.precision_score(y_test, predicted)
print("Recall (all 1s predicted right):", round(recall,2))
print("Precision (confidence when predicting a 1):", round(precision,2))
print("Detail:")
print(metrics.classification_report(y_test, predicted, target_names=[str(i) for i in np.unique(y_test)]))

##############################################################

classes = np.unique(y_test)
fig, ax = plt.subplots(figsize=(10, 6))
cm = metrics.confusion_matrix(y_test, predicted, labels=classes)
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False,annot_kws={"size": 16})
ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
ax.set_title('Confusion matrix',fontsize= 18)
ax.set_yticklabels(labels=classes, rotation=0,fontsize=15)
ax.set_xticklabels(labels=classes, rotation=0,fontsize=15)
plt.show()
#############################################################


# # SVM Classifier

# In[118]:


# LogisticRegression
# Classification and ROC analysis
# Run classifier with cross-validation and plot ROC curves

cv = StratifiedKFold(n_splits=10)
#classifier = svm.SVC(kernel="linear", probability=True)
classifier_SVC = SVC(kernel='rbf',probability=True , class_weight = 'balanced')
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(10, 6))
for i, (train, test) in enumerate(cv.split(X_cv_nmp, y_cv_nmp)):
    classifier_SVC.fit(X_cv_nmp[train], y_cv_nmp[train])
    viz = RocCurveDisplay.from_estimator(
        classifier_SVC,
        X_cv_nmp[test],
        y_cv_nmp[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic of SVM",
)
ax.set_title('Receiver operating characteristic of SVM',fontsize= 18)
ax.legend(loc="lower right")
plt.show()


# In[119]:


from sklearn.model_selection import cross_validate

# train with 10 fold corss validation
#classifier_SVC_cv = SVC(kernel='rbf',probability=True , class_weight = 'balanced')
classifier_SVC = SVC(kernel='rbf',probability=True , class_weight = 'balanced')
scoring=['accuracy','roc_auc','recall','precision','f1']
scores_SVC_cv = cross_validate(classifier_SVC,X_cv_nmp,y_cv_nmp,cv=10,scoring=scoring)

#print(sorted(scores_SVC_cv.keys()))
#print(scores_SVC_cv['test_recall_macro'])
print("10 fold Accuracy: %0.2f (± %0.2f)" % (scores_SVC_cv['test_accuracy'].mean(),scores_SVC_cv['test_accuracy'].std()*2))
print("10 fold AUC: %0.2f (± %0.2f)" % (scores_SVC_cv['test_roc_auc'].mean(),scores_SVC_cv['test_roc_auc'].std()*2))
print("10 fold Recall: %0.2f (± %0.2f)" % (scores_SVC_cv['test_recall'].mean(),scores_SVC_cv['test_recall'].std()*2))
print("10 fold Precision: %0.2f (± %0.2f)" % (scores_SVC_cv['test_precision'].mean(),scores_SVC_cv['test_precision'].std()*2))
print("10 fold f1: %0.2f (± %0.2f)" % (scores_SVC_cv['test_f1'].mean(),scores_SVC_cv['test_f1'].std()*2))


# In[120]:


#############################################################

## train
classifier_SVC.fit(X_train, y_train)
## test
predicted_prob_SVC = classifier_SVC.predict_proba(X_test)[:,1]
predicted = classifier_SVC.predict(X_test)

#############################################################

## Accuray e AUC
accuracy = metrics.accuracy_score(y_test, predicted)
auc_test = metrics.roc_auc_score(y_test, predicted_prob_SVC)
print("Accuracy (overall correct predictions):",  round(accuracy,2))
print("Auc:", round(auc_test,2))
    
## Precision e Recall
recall = metrics.recall_score(y_test, predicted)
precision = metrics.precision_score(y_test, predicted)
print("Recall (all 1s predicted right):", round(recall,2))
print("Precision (confidence when predicting a 1):", round(precision,2))
print("Detail:")
print(metrics.classification_report(y_test, predicted, target_names=[str(i) for i in np.unique(y_test)]))

##############################################################

classes = np.unique(y_test)
fig, ax = plt.subplots(figsize=(10, 6))
cm = metrics.confusion_matrix(y_test, predicted, labels=classes)
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False,annot_kws={"size": 16})
ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
ax.set_title('Confusion matrix',fontsize= 18)
ax.set_yticklabels(labels=classes, rotation=0,fontsize=15)
ax.set_xticklabels(labels=classes, rotation=0,fontsize=15)
plt.show()


##############################################################


# # Decision Tree Classifier

# In[121]:


# DecisionTreeClassifier
# Classification and ROC analysis
# Run classifier with cross-validation and plot ROC curves

cv = StratifiedKFold(n_splits=10)
#classifier = svm.SVC(kernel="linear", probability=True)
classifier_DT = DecisionTreeClassifier(max_depth=6)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(10, 6))
for i, (train, test) in enumerate(cv.split(X_cv_nmp, y_cv_nmp)):
    classifier_DT.fit(X_cv_nmp[train], y_cv_nmp[train])
    viz = RocCurveDisplay.from_estimator(
        classifier_DT,
        X_cv_nmp[test],
        y_cv_nmp[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic of Decision Tree",
)
ax.set_title('Receiver operating characteristic of Decision Tree',fontsize= 18)
ax.legend(loc="lower right")
plt.show()


# In[122]:


from sklearn.model_selection import cross_validate

# train with 10 fold corss validation
#classifier_DT_cv = DecisionTreeClassifier(max_depth=6)
classifier_DT = DecisionTreeClassifier(max_depth=6)
scoring=['accuracy','roc_auc','recall','precision','f1']
scores_DT_cv = cross_validate(classifier_DT,X_cv_nmp,y_cv_nmp,cv=10,scoring=scoring)

#print(sorted(scores_DT_cv.keys()))
#print(scores_DT_cv['test_recall_macro'])
print("10 fold Accuracy: %0.2f (± %0.2f)" % (scores_DT_cv['test_accuracy'].mean(),scores_DT_cv['test_accuracy'].std()*2))
print("10 fold AUC: %0.2f (± %0.2f)" % (scores_DT_cv['test_roc_auc'].mean(),scores_DT_cv['test_roc_auc'].std()*2))
print("10 fold Recall: %0.2f (± %0.2f)" % (scores_DT_cv['test_recall'].mean(),scores_DT_cv['test_recall'].std()*2))
print("10 fold Precision: %0.2f (± %0.2f)" % (scores_DT_cv['test_precision'].mean(),scores_DT_cv['test_precision'].std()*2))
print("10 fold f1: %0.2f (± %0.2f)" % (scores_DT_cv['test_f1'].mean(),scores_DT_cv['test_f1'].std()*2))


# In[123]:


#############################################################

## train
classifier_DT.fit(X_train, y_train)
## test
predicted_prob_DT = classifier_DT.predict_proba(X_test)[:,1]
predicted = classifier_DT.predict(X_test)

#############################################################

## Accuray e AUC
accuracy = metrics.accuracy_score(y_test, predicted)
auc_test = metrics.roc_auc_score(y_test, predicted_prob_DT)
print("Accuracy (overall correct predictions):",  round(accuracy,2))
print("Auc:", round(auc_test,2))
    
## Precision e Recall
recall = metrics.recall_score(y_test, predicted)
precision = metrics.precision_score(y_test, predicted)
print("Recall (all 1s predicted right):", round(recall,2))
print("Precision (confidence when predicting a 1):", round(precision,2))
print("Detail:")
print(metrics.classification_report(y_test, predicted, target_names=[str(i) for i in np.unique(y_test)]))

##############################################################

classes = np.unique(y_test)
fig, ax = plt.subplots(figsize=(10, 6))
cm = metrics.confusion_matrix(y_test, predicted, labels=classes)
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False,annot_kws={"size": 16})
ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
ax.set_title('Confusion matrix',fontsize= 18)
ax.set_yticklabels(labels=classes, rotation=0,fontsize=15)
ax.set_xticklabels(labels=classes, rotation=0,fontsize=15)
plt.show()


##############################################################


# In[124]:


from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree

#feature name
feature_names = X_train.columns.values.tolist()

#class name
class_names=['Control','LAA']

text_representation = tree.export_text(classifier_DT,feature_names=feature_names)
print(text_representation)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(classifier_DT, 
                   feature_names=feature_names,  
                   class_names=class_names,
                   filled=True,
                   max_depth=3,
                   fontsize=10)


# # Random Forest Classifier

# In[125]:


# RandomForestClassifier
# Classification and ROC analysis
# Run classifier with cross-validation and plot ROC curves

cv = StratifiedKFold(n_splits=10)
#classifier = svm.SVC(kernel="linear", probability=True)
classifier_RF = RandomForestClassifier()
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(10, 6))
for i, (train, test) in enumerate(cv.split(X_cv_nmp, y_cv_nmp)):
    classifier_RF.fit(X_cv_nmp[train], y_cv_nmp[train])
    viz = RocCurveDisplay.from_estimator(
        classifier_RF,
        X_cv_nmp[test],
        y_cv_nmp[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic of Random Forest",
)
ax.set_title('Receiver operating characteristic of Random Forest',fontsize= 18)
ax.legend(loc="lower right")
plt.show()


# In[126]:


from sklearn.model_selection import cross_validate

# train with 10 fold corss validation
# classifier_RF_cv = RandomForestClassifier()
classifier_RF = RandomForestClassifier()
scoring=['accuracy','roc_auc','recall','precision','f1']
scores_RF_cv = cross_validate(classifier_RF,X_cv_nmp,y_cv_nmp,cv=10,scoring=scoring)

#print(sorted(scores_RF_cv.keys()))
#print(scores_RF_cv['test_recall_macro'])
print("10 fold Accuracy: %0.2f (± %0.2f)" % (scores_RF_cv['test_accuracy'].mean(),scores_RF_cv['test_accuracy'].std()*2))
print("10 fold AUC: %0.2f (± %0.2f)" % (scores_RF_cv['test_roc_auc'].mean(),scores_RF_cv['test_roc_auc'].std()*2))
print("10 fold Recall: %0.2f (± %0.2f)" % (scores_RF_cv['test_recall'].mean(),scores_RF_cv['test_recall'].std()*2))
print("10 fold Precision: %0.2f (± %0.2f)" % (scores_RF_cv['test_precision'].mean(),scores_RF_cv['test_precision'].std()*2))
print("10 fold f1: %0.2f (± %0.2f)" % (scores_RF_cv['test_f1'].mean(),scores_RF_cv['test_f1'].std()*2))


# In[127]:


#############################################################

## train
classifier_RF.fit(X_train, y_train)
## test
predicted_prob_RF = classifier_RF.predict_proba(X_test)[:,1]
predicted = classifier_RF.predict(X_test)

#############################################################

## Accuray e AUC
accuracy = metrics.accuracy_score(y_test, predicted)
auc_test = metrics.roc_auc_score(y_test, predicted_prob_RF)
print("Accuracy (overall correct predictions):",  round(accuracy,2))
print("Auc:", round(auc_test,2))
    
## Precision e Recall
recall = metrics.recall_score(y_test, predicted)
precision = metrics.precision_score(y_test, predicted)
print("Recall (all 1s predicted right):", round(recall,2))
print("Precision (confidence when predicting a 1):", round(precision,2))
print("Detail:")
print(metrics.classification_report(y_test, predicted, target_names=[str(i) for i in np.unique(y_test)]))

##############################################################

classes = np.unique(y_test)
fig, ax = plt.subplots(figsize=(10, 6))
cm = metrics.confusion_matrix(y_test, predicted, labels=classes)
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False,annot_kws={"size": 16})
ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
ax.set_title('Confusion matrix',fontsize= 18)
ax.set_yticklabels(labels=classes, rotation=0,fontsize=15)
ax.set_xticklabels(labels=classes, rotation=0,fontsize=15)
plt.show()


##############################################################


# In[128]:


# Show all columns as list
feature_names = X_train.columns.values.tolist()

import time
import numpy as np

plt.figure(figsize=(10, 8))
plt.title(fontsize=15,label="Feature Importance")
start_time = time.time()
importances = classifier_RF.feature_importances_
std = np.std([tree.feature_importances_ for tree in classifier_RF.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

import pandas as pd

forest_importances = pd.Series(importances, index=feature_names)
forest_importances.nlargest(25).plot(kind='barh')
#forest_importances.plot(kind='barh')
# fig, ax = plt.subplots()
#forest_importances.plot.bar(yerr=std, ax=ax)
# forest_importances.set_title("Feature importances using MDI")
# forest_importances.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()


# # XGB Classifier

# In[129]:


#list(X_train.columns.values)


# In[130]:


# RandomcolumnsestClassifier
# Classification and ROC analysis
# Run classifier with cross-validation and plot ROC curves

cv = StratifiedKFold(n_splits=10)
#classifier = svm.SVC(kernel="linear", probability=True)
classifier_XGB = XGBClassifier(objective='binary:logistic',
                          booster='gbtree',
                          eval_metric='auc',
                          tree_method='hist',
                          grow_policy='lossguide',
                          use_label_encoder=None)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(10, 6))
for i, (train, test) in enumerate(cv.split(X_cv_nmp, y_cv_nmp)):
    classifier_XGB.fit(X_cv_nmp[train], y_cv_nmp[train])
    viz = RocCurveDisplay.from_estimator(
        classifier_XGB,
        X_cv_nmp[test],
        y_cv_nmp[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic of XGBoost",
)
ax.set_title('Receiver operating characteristic of XGBoost',fontsize= 18)
ax.legend(loc="lower right")
plt.show()


# In[131]:


from sklearn.model_selection import cross_validate

# train with 10 fold corss validation
# classifier_XGB_cv = XGBClassifier(objective='binary:logistic',
#                           booster='gbtree',
#                           eval_metric='auc',
#                           tree_method='hist',
#                           grow_policy='lossguide',
#                           use_label_encoder=None)

classifier_XGB = XGBClassifier(objective='binary:logistic',
                          booster='gbtree',
                          eval_metric='auc',
                          tree_method='hist',
                          grow_policy='lossguide',
                          use_label_encoder=None)

scoring=['accuracy','roc_auc','recall','precision','f1']

scores_XGB_cv = cross_validate(classifier_XGB,X_cv_nmp,y_cv_nmp,cv=10,scoring=scoring)

#print(sorted(scores_XGB_cv.keys()))
#print(scores_XGB_cv['test_recall_macro'])
print("10 fold Accuracy: %0.2f (± %0.2f)" % (scores_XGB_cv['test_accuracy'].mean(),scores_XGB_cv['test_accuracy'].std()*2))
print("10 fold AUC: %0.2f (± %0.2f)" % (scores_XGB_cv['test_roc_auc'].mean(),scores_XGB_cv['test_roc_auc'].std()*2))
print("10 fold Recall: %0.2f (± %0.2f)" % (scores_XGB_cv['test_recall'].mean(),scores_XGB_cv['test_recall'].std()*2))
print("10 fold Precision: %0.2f (± %0.2f)" % (scores_XGB_cv['test_precision'].mean(),scores_XGB_cv['test_precision'].std()*2))
print("10 fold f1: %0.2f (± %0.2f)" % (scores_XGB_cv['test_f1'].mean(),scores_XGB_cv['test_f1'].std()*2))


# In[132]:


#############################################################

## train
classifier_XGB.fit(X_train, y_train)
## test
predicted_prob_XGB = classifier_XGB.predict_proba(X_test)[:,1]
predicted = classifier_XGB.predict(X_test)

#############################################################

## Accuray e AUC
accuracy = metrics.accuracy_score(y_test, predicted)
auc_test = metrics.roc_auc_score(y_test, predicted_prob_XGB)
print("Accuracy (overall correct predictions):",round(accuracy,2))
print("Auc:", round(auc_test,2))
    
## Precision e Recall
recall = metrics.recall_score(y_test, predicted)
precision = metrics.precision_score(y_test, predicted)
print("Recall (all 1s predicted right):", round(recall,2))
print("Precision (confidence when predicting a 1):", round(precision,2))
print("Detail:")
print(metrics.classification_report(y_test, predicted, target_names=[str(i) for i in np.unique(y_test)]))

##############################################################

classes = np.unique(y_test)
fig, ax = plt.subplots(figsize=(10, 6))
cm = metrics.confusion_matrix(y_test, predicted, labels=classes)
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False,annot_kws={"size": 16})
ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
ax.set_title('Confusion matrix',fontsize= 18)
ax.set_yticklabels(labels=classes, rotation=0,fontsize=15)
ax.set_xticklabels(labels=classes, rotation=0,fontsize=15)
plt.show()


##############################################################


# # Gradient Boost

# In[133]:


# Gradient Boost
# Classification and ROC analysis
# Run classifier with cross-validation and plot ROC curves

cv = StratifiedKFold(n_splits=10)
#classifier = svm.SVC(kernel="linear", probability=True)
classifier_GBC = GradientBoostingClassifier()
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(10, 6))
for i, (train, test) in enumerate(cv.split(X_cv_nmp, y_cv_nmp)):
    classifier_GBC.fit(X_cv_nmp[train], y_cv_nmp[train])
    viz = RocCurveDisplay.from_estimator(
        classifier_GBC,
        X_cv_nmp[test],
        y_cv_nmp[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic of GradientBoost",
)
ax.set_title('Receiver operating characteristic of GradientBoost',fontsize= 18)
ax.legend(loc="lower right")
plt.show()


# In[134]:


from sklearn.model_selection import cross_validate

# train with 10 fold corss validation
# classifier_GBC_cv = GradientBoostingClassifier()
classifier_GBC = GradientBoostingClassifier()
scoring=['accuracy','roc_auc','recall','precision','f1']
scores_GBC_cv = cross_validate(classifier_GBC,X_cv_nmp,y_cv_nmp,cv=10,scoring=scoring)

#print(sorted(scores_GBC_cv.keys()))
#print(scores_GBC_cv['test_recall_macro'])
print("10 fold Accuracy: %0.2f (± %0.2f)" % (scores_GBC_cv['test_accuracy'].mean(),scores_GBC_cv['test_accuracy'].std()*2))
print("10 fold AUC: %0.2f (± %0.2f)" % (scores_GBC_cv['test_roc_auc'].mean(),scores_GBC_cv['test_roc_auc'].std()*2))
print("10 fold Recall: %0.2f (± %0.2f)" % (scores_GBC_cv['test_recall'].mean(),scores_GBC_cv['test_recall'].std()*2))
print("10 fold Precision: %0.2f (± %0.2f)" % (scores_GBC_cv['test_precision'].mean(),scores_GBC_cv['test_precision'].std()*2))
print("10 fold f1: %0.2f (± %0.2f)" % (scores_GBC_cv['test_f1'].mean(),scores_GBC_cv['test_f1'].std()*2))


# In[135]:


#############################################################

## train
classifier_GBC.fit(X_train, y_train)

## test
predicted_prob_GBC = classifier_GBC.predict_proba(X_test)[:,1]
predicted = classifier_GBC.predict(X_test)

#############################################################

## Accuray e AUC
accuracy = metrics.accuracy_score(y_test, predicted)
auc_test = metrics.roc_auc_score(y_test, predicted_prob_GBC)
print("Accuracy (overall correct predictions):",  round(accuracy,2))
print("Auc:", round(auc_test,2))
    
## Precision e Recall
recall = metrics.recall_score(y_test, predicted)
precision = metrics.precision_score(y_test, predicted)
print("Recall (all 1s predicted right):", round(recall,2))
print("Precision (confidence when predicting a 1):", round(precision,2))
print("Detail:")
print(metrics.classification_report(y_test, predicted, target_names=[str(i) for i in np.unique(y_test)]))

##############################################################

classes = np.unique(y_test)
fig, ax = plt.subplots(figsize=(10, 6))
cm = metrics.confusion_matrix(y_test, predicted, labels=classes)
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False,annot_kws={"size": 16})
ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
ax.set_title('Confusion matrix',fontsize= 18)
ax.set_yticklabels(labels=classes, rotation=0,fontsize=15)
ax.set_xticklabels(labels=classes, rotation=0,fontsize=15)
plt.show()


##############################################################


# In[136]:


def roc_curve_and_score(y_test, pred_proba):
    fpr, tpr, _ = roc_curve(y_test.ravel(), pred_proba.ravel())
    roc_auc = roc_auc_score(y_test.ravel(), pred_proba.ravel())
    return fpr, tpr, roc_auc

plt.figure(figsize=(5, 5))
#plt.rcParams.update({'font.size': 14})
plt.grid()

fpr, tpr, roc_auc = roc_curve_and_score(y_test, predicted_prob_LR)
plt.plot(fpr, tpr, color='blue', lw=2,
         label='Logistic Regression (area={0:.3f})'.format(roc_auc))
fpr, tpr, roc_auc = roc_curve_and_score(y_test, predicted_prob_SVC)
plt.plot(fpr, tpr, color='green', lw=2,
         label='SVM (area={0:.3f})'.format(roc_auc))
fpr, tpr, roc_auc = roc_curve_and_score(y_test, predicted_prob_DT)
plt.plot(fpr, tpr, color='red', lw=2,
         label='Decision Tree (area={0:.3f})'.format(roc_auc))
fpr, tpr, roc_auc = roc_curve_and_score(y_test, predicted_prob_RF)
plt.plot(fpr, tpr, color='black', lw=2,
         label='Random Forest (area={0:.3f})'.format(roc_auc))
fpr, tpr, roc_auc = roc_curve_and_score(y_test, predicted_prob_XGB)
plt.plot(fpr, tpr, color='orange', lw=2,
         label='XGBoost (area={0:.3f})'.format(roc_auc))
fpr, tpr, roc_auc = roc_curve_and_score(y_test, predicted_prob_GBC)
plt.plot(fpr, tpr, color='brown', lw=2,
         label='Gradient Boost (area={0:.3f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.legend(loc="lower right")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.suptitle('ROC curves of models (Clinical+Metabolomics data)',fontsize=15)
plt.savefig("./data/plot.Clinical+Metabo.png", dpi=300)
plt.savefig("./data/plot.Clinical+Metabo.pdf")
plt.show()


# # RFE (Recursive Feature Elimination) 

# In[58]:


from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV


# In[59]:


X_train_rfe=X_train
X_test_rfe=X_test
y_train_rfe=y_train
y_test_rfe=y_test


# In[60]:


cv_rfe = StratifiedKFold(n_splits=10)


# ## RFE (Logistic Regression)

# In[61]:


selector_LR = RFECV(LogisticRegression(max_iter=3000), cv=cv_rfe,scoring="roc_auc")
selector_LR = selector_LR.fit(X_train_rfe, y_train_rfe)
g_scores_LR = selector_LR.grid_scores_
avg_LR = np.average(g_scores_LR, axis=1)

rfe_kept_LR = pd.DataFrame({'columns': X_train_rfe.columns,'Kept': selector_LR.support_})
rfe_kept_LR_select=rfe_kept_LR[rfe_kept_LR["Kept"]== True]
kept_LR=rfe_kept_LR_select['columns'].array

len(rfe_kept_LR_select)
print("Optimal AUC of features is : %f" % avg_LR.max() )
print("Optimal number of features in LR is: %d" % selector_LR.n_features_)


# In[62]:


g_scores_LR


# In[63]:


kept_LR


# In[64]:


g_scores_LR_df=pd.DataFrame(g_scores_LR)
g_scores_LR_df.to_csv("./new-data-220328/gridscores/g_scores_LR.txt",index=False, sep="\t")


# In[65]:


import numpy as np
import matplotlib.pyplot as plt



# data to be plotted
x = np.arange(0, len(avg_LR))
y = avg_LR

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(x,y)
ax.title.set_text('RFECV for Logistic Regression')
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "AUC={:.3f} \n Num. of feature ={:.0f} ".format(ymax,xmax+1)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="k", lw=0.5)
    #arrowprops=dict(facecolor='black', shrink=0.01)
    kw = dict(xycoords='data',textcoords="axes fraction",
               bbox=bbox_props, ha="right", va="center")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.95, 0.1), **kw,fontsize=15)
    ax.axvline(x=xmax, color='r', label='test lines',linestyle='--')
annot_max(x,y)
plt.yticks(np.arange(0.5, 0.9, step=0.05))
plt.xlabel('Number of features')
plt.ylabel('AUC')
plt.show()


# ## RFE (SVM)

# In[66]:


selector_SVC = RFECV(SVC(kernel='linear'), cv=cv_rfe,scoring="roc_auc")
selector_SVC = selector_SVC.fit(X_train_rfe, y_train_rfe)
g_scores_SVC = selector_SVC.grid_scores_
avg_SVC = np.average(g_scores_SVC, axis=1)

rfe_kept_SVC = pd.DataFrame({'columns': X_train_rfe.columns,'Kept': selector_SVC.support_})
rfe_kept_SVC_select=rfe_kept_SVC[rfe_kept_SVC["Kept"]== True]
kept_SVC=rfe_kept_SVC_select['columns'].array

len(rfe_kept_SVC_select)
print("Optimal AUC of features is : %f" % avg_SVC.max() )
print("Optimal number of features in SVC is: %d" % selector_SVC.n_features_)


# In[67]:


kept_SVC


# In[68]:


g_scores_SVC_df=pd.DataFrame(g_scores_SVC)
g_scores_SVC_df.to_csv("./new-data-220328/gridscores/g_scores_SVC.txt",index=False, sep="\t")


# In[69]:


import numpy as np
import matplotlib.pyplot as plt



# data to be plotted
x = np.arange(0, len(avg_SVC))
y = avg_SVC

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(x,y)
ax.title.set_text('RFECV for SVM')
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "AUC={:.3f} \n Num. of feature ={:.0f} ".format(ymax,xmax+1)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="k", lw=0.5)
    #arrowprops=dict(facecolor='black', shrink=0.01)
    kw = dict(xycoords='data',textcoords="axes fraction",
               bbox=bbox_props, ha="right", va="center")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.95, 0.1), **kw,fontsize=15)
    ax.axvline(x=xmax, color='r', label='test lines',linestyle='--')
annot_max(x,y)
plt.yticks(np.arange(0.5, 0.9, step=0.05))
plt.xlabel('Number of features')
plt.ylabel('AUC')
plt.show()


# ## RFE (Decision Tree)

# In[70]:


selector_DT = RFECV(DecisionTreeClassifier(criterion='gini'), cv=cv_rfe,scoring="roc_auc")
selector_DT = selector_DT.fit(X_train_rfe, y_train_rfe)
g_scores_DT = selector_DT.grid_scores_
avg_DT = np.average(g_scores_DT, axis=1)

rfe_kept_DT = pd.DataFrame({'columns': X_train_rfe.columns,'Kept': selector_DT.support_})
rfe_kept_DT_select=rfe_kept_DT[rfe_kept_DT["Kept"]== True]
kept_DT=rfe_kept_DT_select['columns'].array

len(rfe_kept_DT_select)
print("Optimal AUC of features is : %f" % avg_DT.max() )
print("Optimal number of features in DT is: %d" % selector_DT.n_features_)


# In[71]:


kept_DT


# In[72]:


g_scores_DT_df=pd.DataFrame(g_scores_DT)
g_scores_DT_df.to_csv("./new-data-220328/gridscores/g_scores_DT.txt",index=False, sep="\t")


# In[73]:


import numpy as np
import matplotlib.pyplot as plt



# data to be plotted
x = np.arange(0, len(avg_DT))
y = avg_DT

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(x,y)
ax.title.set_text('RFECV for Decision Tree')
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "AUC={:.3f} \n Num. of feature ={:.0f} ".format(ymax,xmax+1)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="k", lw=0.5)
    #arrowprops=dict(facecolor='black', shrink=0.01)
    kw = dict(xycoords='data',textcoords="axes fraction",
               bbox=bbox_props, ha="right", va="center")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.95, 0.1), **kw,fontsize=15)
    ax.axvline(x=xmax, color='r', label='test lines',linestyle='--')
annot_max(x,y)
plt.yticks(np.arange(0.5, 0.9, step=0.05))
plt.xlabel('Number of features')
plt.ylabel('AUC')
plt.show()


# ## RFE (Random Forest)

# In[74]:


selector_RF = RFECV(RandomForestClassifier(), cv=cv_rfe,scoring="roc_auc")
selector_RF = selector_RF.fit(X_train_rfe, y_train_rfe)
g_scores_RF = selector_RF.grid_scores_
avg_RF = np.average(g_scores_RF, axis=1)

rfe_kept_RF = pd.DataFrame({'columns': X_train_rfe.columns,'Kept': selector_RF.support_})
rfe_kept_RF_select=rfe_kept_RF[rfe_kept_RF["Kept"]== True]
kept_RF=rfe_kept_RF_select['columns'].array

len(rfe_kept_RF_select)
print("Optimal AUC of features is : %f" % avg_RF.max() )
print("Optimal number of features in RF is: %d" % selector_RF.n_features_)


# In[75]:


kept_RF


# In[76]:


g_scores_RF_df=pd.DataFrame(g_scores_RF)
g_scores_RF_df.to_csv("./new-data-220328/gridscores/g_scores_RF.txt",index=False, sep="\t")


# In[77]:


import numpy as np
import matplotlib.pyplot as plt



# data to be plotted
x = np.arange(0, len(avg_RF))
y = avg_RF

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(x,y)
ax.title.set_text('RFECV for Random Forest')
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "AUC={:.3f} \n Num. of feature ={:.0f} ".format(ymax,xmax+1)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="k", lw=0.5)
    #arrowprops=dict(facecolor='black', shrink=0.01)
    kw = dict(xycoords='data',textcoords="axes fraction",
               bbox=bbox_props, ha="right", va="center")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.95, 0.1), **kw,fontsize=15)
    ax.axvline(x=xmax, color='r', label='test lines',linestyle='--')
annot_max(x,y)
plt.yticks(np.arange(0.5, 0.9, step=0.05))
plt.xlabel('Number of features')
plt.ylabel('AUC')
plt.show()


# ## RFE (XGB)

# In[78]:


selector_XGB = RFECV(XGBClassifier(eval_metric='auc'), cv=cv_rfe,scoring="roc_auc")
selector_XGB = selector_XGB.fit(X_train_rfe, y_train_rfe)
g_scores_XGB = selector_XGB.grid_scores_
avg_XGB = np.average(g_scores_XGB, axis=1)


rfe_kept_XGB = pd.DataFrame({'columns': X_train_rfe.columns,'Kept': selector_XGB.support_})
rfe_kept_XGB_select=rfe_kept_XGB[rfe_kept_XGB["Kept"]== True]
kept_XGB=rfe_kept_XGB_select['columns'].array

len(rfe_kept_XGB_select)
print("Optimal AUC of features is : %f" % avg_XGB.max() )
print("Optimal number of features in XGB is: %d" % selector_XGB.n_features_)


# In[79]:


kept_XGB


# In[80]:


g_scores_XGB_df=pd.DataFrame(g_scores_XGB)
g_scores_XGB_df.to_csv("./new-data-220328/gridscores/g_scores_XGB.txt",index=False, sep="\t")


# In[81]:


import numpy as np
import matplotlib.pyplot as plt



# data to be plotted
x = np.arange(0, len(avg_XGB))
y = avg_XGB

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(x,y)
ax.title.set_text('RFECV for XGBoost')
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "AUC={:.3f} \n Num. of feature ={:.0f} ".format(ymax,xmax+1)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="k", lw=0.5)
    #arrowprops=dict(facecolor='black', shrink=0.01)
    kw = dict(xycoords='data',textcoords="axes fraction",
               bbox=bbox_props, ha="right", va="center")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.95, 0.1), **kw,fontsize=15)
    ax.axvline(x=xmax, color='r', label='test lines',linestyle='--')
annot_max(x,y)
plt.yticks(np.arange(0.5, 0.9, step=0.05))
plt.xlabel('Number of features')
plt.ylabel('AUC')
plt.show()


# ## RFE (GradientBoosting)

# In[82]:


selector_GBC = RFECV(GradientBoostingClassifier(), cv=cv_rfe,scoring="roc_auc")
selector_GBC = selector_GBC.fit(X_train_rfe, y_train_rfe)
g_scores_GBC = selector_GBC.grid_scores_
avg_GBC = np.average(g_scores_GBC, axis=1)

rfe_kept_GBC = pd.DataFrame({'columns': X_train_rfe.columns,'Kept': selector_GBC.support_})
rfe_kept_GBC_select=rfe_kept_GBC[rfe_kept_GBC["Kept"]== True]
kept_GBC=rfe_kept_GBC_select['columns'].array

len(rfe_kept_GBC_select)
print("Optimal AUC of features is : %f" % avg_GBC.max() )
print("Optimal number of features in GBC is: %d" % selector_GBC.n_features_)


# In[83]:


kept_GBC


# In[84]:


g_scores_GBC_df=pd.DataFrame(g_scores_GBC)
g_scores_GBC_df.to_csv("./new-data-220328/gridscores/g_scores_GBC.txt",index=False, sep="\t")


# In[85]:


import numpy as np
import matplotlib.pyplot as plt



# data to be plotted
x = np.arange(0, len(avg_GBC))
y = avg_GBC

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(x,y)
ax.title.set_text('RFECV for GradientBoosting')
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "AUC={:.3f} \n Num. of feature ={:.0f} ".format(ymax,xmax+1)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="k", lw=0.5)
    #arrowprops=dict(facecolor='black', shrink=0.01)
    kw = dict(xycoords='data',textcoords="axes fraction",
               bbox=bbox_props, ha="right", va="center")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.95, 0.1), **kw,fontsize=15)
    ax.axvline(x=xmax, color='r', label='test lines',linestyle='--')
annot_max(x,y)
plt.yticks(np.arange(0.5, 0.9, step=0.05))
plt.xlabel('Number of features')
plt.ylabel('AUC')
plt.show()


# In[104]:


from venn import venn
from venn import pseudovenn

get_ipython().run_line_magic('matplotlib', 'inline')

set_LR = set(kept_LR)
set_SVC = set(kept_SVC)
set_DT = set(kept_DT)
set_RF = set(kept_RF)
set_XGB = set(kept_XGB)
set_GBC = set(kept_GBC)


feature_plot={
    'Logistic Regression':set_LR,
    'SVM Classifier':set_SVC,
    'Decision Tree':set_DT,
    'Random Forest':set_RF,
    'Xgboost':set_XGB,
    'Gradient Boost':set_GBC
}

pseudovenn(feature_plot,fontsize=15,figsize=(18,18),hint_hidden=False)


# In[105]:


feature_6_models=list(set(set_LR) & set(set_SVC) & set(set_DT) & set(set_RF) & set(set_XGB) & set(set_GBC))


# In[106]:


len(feature_6_models)


# In[107]:


feature_6_models


# # Save the model feature parameter (Features on Paper)

# In[137]:


set_LR_save={'Ala','Alcohol','Asn','BMI','BW', 'C0', 'C10', 'C12DC', 'C141', 'C142OH', 'C182', 'C2', 'C3', 'C4', 'C41', 'C5','C6C41DC', 'C8', 'CHOL', 'Cit', 'Creatinine', 'DBP', 'DM', 'DM_drug', 'HDL', 'HT', 'HTN_drug', 'Kynurenine', 'LDL', 'Lipid_drug', 'Lys', 'PCaaC300', 'PCaaC360', 'PCaaC364', 'PCaaC365', 'PCaaC404', 'PCaaC406', 'PCaaC422', 'PCaaC424', 'PCaeC302','PCaeC342','PCaeC360', 'PCaeC362','PCaeC383','PCaeC405','PCaeC424','Phe','Pro','SMC180','SMC181','SMC241','SMOHC161','SMOHC221','SMOHC222','Sarcosine','Ser','Smoking','Thr','Trp','age','lysoPCaC260','lysoPCaC281'}
len(set_LR_save)


# In[138]:


set_SVC_save={'Ala', 'Alcohol', 'Asn', 'BMI', 'BW', 'C0', 'C10', 'C102', 'C12DC', 'C14', 'C141', 'C142', 'C142OH', 'C161OH', 'C181OH', 'C182', 'C2', 'C3', 'C4', 'C41', 'C5', 'C6C41DC', 'C7DC', 'C8', 'C9', 'CHOL', 'Creatinine', 'DBP', 'DM', 'DM_drug', 'HDL', 'HT', 'HTN_drug', 'Kynurenine', 'LDL', 'Lipid_drug', 'Lys', 'Orn', 'PCaaC300', 'PCaaC343', 'PCaaC344', 'PCaaC360', 'PCaaC361', 'PCaaC362', 'PCaaC363', 'PCaaC364', 'PCaaC365', 'PCaaC403', 'PCaaC404', 'PCaaC406', 'PCaaC422', 'PCaaC424', 'PCaeC302', 'PCaeC322', 'PCaeC341','PCaeC360', 'PCaeC364', 'PCaeC380', 'PCaeC382', 'PCaeC383', 'PCaeC401', 'PCaeC405', 'PCaeC424', 'PCaeC444', 'Phe', 'Pro', 'SMC161','SMC180', 'SMC261', 'SMOHC161', 'SMOHC221','SMOHC222', 'Sarcosine', 'Ser', 'Smoking', 'TG', 'Thr', 'Trp', 'age', 'lysoPCaC170', 'lysoPCaC204', 'lysoPCaC240', 'lysoPCaC281'}
len(set_SVC_save)


# In[139]:


set_DT_save={'Arg', 'BMI', 'C182', 'C2', 'C4', 'C9', 'HTN_drug', 'HsCRP', 'Kynurenine', 'PCaeC361', 'Smoking', 'Trp', 'Val'}
len(set_DT_save)


# In[140]:


set_RF_save={'AcSugar', 'Asp', 'BMI', 'BW', 'C0', 'C10','C101', 'C12', 'C121', 'C141', 'C161', 'C182', 'C3', 'C4', 'C5', 'C7DC', 'C8', 'C9', 'CHOL', 'Cit', 'Creatinine', 'Creatinine_MS', 'DBP', 'DM', 'DM_drug', 'Glu', 'HDL', 'HTN_drug', 'Hip', 'Homocysteine', 'HsCRP', 'Ile', 'Kynurenine', 'LDL', 'Leu', 'Lipid_drug', 'Lys', 'MBP', 'Met', 'PCaaC240', 'PCaaC281', 'PCaaC323', 'PCaaC343', 'PCaaC344', 'PCaaC361', 'PCaaC364', 'PCaaC366', 'PCaaC383','PCaaC384', 'PCaaC402', 'PCaaC406', 'PCaaC421', 'PCaaC422', 'PCaaC425', 'PCaaC426', 'PCaeC300', 'PCaeC302', 'PCaeC321', 'PCaeC322', 'PCaeC340', 'PCaeC342', 'PCaeC343', 'PCaeC362', 'PCaeC363', 'PCaeC380', 'PCaeC386', 'PCaeC420', 'PCaeC445', 'PCaeC446', 'Phe', 'Pro', 'SBP', 'SDMA', 'SMC161', 'SMC181', 'SMC202', 'SMC241', 'SMC260', 'SMC261', 'SMOHC141', 'SMOHC161', 'SMOHC241', 'Sarcosine', 'Ser', 'TG', 'Trp', 'Tyr', 'UA', 'Val', 'age', 'lysoPCaC161', 'lysoPCaC182', 'lysoPCaC203', 'lysoPCaC204', 'lysoPCaC260'}
len(set_RF_save)


# In[141]:


set_XGB_save={'AcSugar', 'Ala', 'Alcohol', 'Arg', 'Asn', 'Asp', 'BMI', 'BW', 'C0', 'C10', 'C101', 'C102', 'C12', 'C121', 'C141', 'C16', 'C161', 'C16OH', 'C18', 'C181', 'C182', 'C2', 'C3', 'C4', 'C51', 'C51DC', 'C6C41DC', 'C7DC', 'C8', 'CHOL', 'Cit', 'Creatinine', 'Creatinine_MS', 'DBP', 'DM', 'Gln', 'Glu', 'Gly', 'HDL', 'HT', 'HTN_drug', 'HeartRate', 'Hip', 'His', 'Homocysteine', 'HsCRP', 'Ile', 'Kynurenine', 'LDL', 'Lipid_drug', 'MBP', 'Met', 'Orn', 'PCaaC281', 'PCaaC300', 'PCaaC323', 'PCaaC341', 'PCaaC342', 'PCaaC360', 'PCaaC364', 'PCaaC365', 'PCaaC366', 'PCaaC383', 'PCaaC384', 'PCaaC385', 'PCaaC403', 'PCaaC404', 'PCaaC406', 'PCaaC422', 'PCaeC300', 'PCaeC322', 'PCaeC340', 'PCaeC342', 'PCaeC343', 'PCaeC360', 'PCaeC361', 'PCaeC363', 'PCaeC380', 'PCaeC382', 'PCaeC385', 'PCaeC386', 'PCaeC420', 'PCaeC421', 'PCaeC423', 'PCaeC446', 'Phe', 'Pro', 'SBP', 'SDMA', 'SMC180', 'SMC181', 'SMC202', 'SMC240', 'SMC241', 'SMC260', 'SMC261', 'SMOHC141', 'SMOHC161', 'SMOHC241', 'Sarcosine', 'Ser', 'Smoking', 'TG', 'Trp', 'UA', 'Val', 'age', 'lysoPCaC161', 'lysoPCaC203', 'lysoPCaC204', 'lysoPCaC240', 'lysoPCaC260', 'lysoPCaC261', 'lysoPCaC281','t4OHPro'}
len(set_XGB_save)


# In[142]:


set_GBC_save={'ADMA', 'AcSugar', 'Ala', 'Arg', 'Asn', 'Asp', 'BH', 'BMI', 'BW', 'C0', 'C10', 'C101', 'C102', 'C12', 'C121', 'C12DC', 'C141', 'C161', 'C161OH', 'C162OH', 'C18', 'C182', 'C2', 'C3', 'C3DCC4OH', 'C3OH', 'C4', 'C5', 'C51', 'C51DC', 'C5DCC6OH', 'C61', 'C7DC', 'C8', 'CHOL', 'Cit', 'Creatinine', 'Creatinine_MS', 'DBP', 'DM', 'DM_drug', 'Gln', 'Glu', 'Gly', 'HDL', 'HT', 'HTN_drug', 'Hip', 'HsCRP', 'Ile', 'Kynurenine', 'LDL', 'Leu', 'Lipid_drug', 'Lys', 'Met', 'Orn', 'PCaaC240', 'PCaaC300', 'PCaaC323', 'PCaaC341', 'PCaaC342', 'PCaaC361', 'PCaaC363', 'PCaaC364', 'PCaaC366', 'PCaaC380', 'PCaaC383', 'PCaaC384', 'PCaaC385', 'PCaaC402', 'PCaaC404', 'PCaaC406', 'PCaeC300', 'PCaeC321', 'PCaeC340', 'PCaeC342', 'PCaeC343', 'PCaeC360', 'PCaeC361', 'PCaeC362', 'PCaeC380', 'PCaeC382', 'PCaeC383', 'PCaeC385', 'PCaeC386', 'PCaeC402', 'PCaeC420', 'PCaeC421', 'PCaeC422', 'PCaeC423', 'PCaeC425', 'PCaeC446', 'Phe', 'Pro', 'SBP', 'SDMA', 'SMC180', 'SMC181', 'SMC202', 'SMC240', 'SMC241', 'SMC260', 'SMC261', 'SMOHC161', 'SMOHC222', 'SMOHC241', 'Sarcosine', 'Ser', 'Smoking', 'TG', 'Trp', 'UA', 'Val', 'age', 'alphaAAA', 'lysoPCaC161', 'lysoPCaC182', 'lysoPCaC203', 'lysoPCaC204', 'lysoPCaC240', 'lysoPCaC260','t4OHPro'}
len(set_GBC_save)


# In[143]:


feature_6_models_save={'Trp', 'Kynurenine', 'BMI', 'C182', 'C4', 'HTN_drug'}
len(feature_6_models_save)


# In[144]:


feature_5_models=list(set(set_LR_save) & set(set_SVC_save) & set(set_RF_save) & set(set_XGB_save) & set(set_GBC_save))
len(feature_5_models)


# In[145]:


#feature_5_models


# In[155]:


from venn import venn
from venn import pseudovenn
import matplotlib

get_ipython().run_line_magic('matplotlib', 'inline')

set_LR_save = set(set_LR_save)
set_SVC_save = set(set_SVC_save)
set_RF_save = set(set_RF_save)
set_XGB_save = set(set_XGB_save)
set_GBC_save = set(set_GBC_save)


feature_plot={
    'Logistic Regression':set_LR_save,
    'SVM Classifier':set_SVC_save,
    'Random Forest':set_RF_save,
    'Xgboost':set_XGB_save,
    'Gradient Boost':set_GBC_save
}

venn(feature_plot,fontsize=15,figsize=(13,13),hint_hidden=False)

matplotlib.pyplot.savefig('./flower.png')
matplotlib.pyplot.savefig('./flower.pdf')


# In[214]:


df_LR_save=pd.DataFrame(set_LR_save)
df_SVC_save=pd.DataFrame(set_SVC_save)
df_DT_save=pd.DataFrame(set_DT_save)
df_RF_save=pd.DataFrame(set_RF_save)
df_XGB_save=pd.DataFrame(set_XGB_save)
df_GBC_save=pd.DataFrame(set_GBC_save)
df_feature_5_models=pd.DataFrame(feature_5_models)


# In[215]:


df_all_save=pd.concat([df_LR_save.reset_index(drop=True), df_SVC_save,df_DT_save,df_RF_save,df_XGB_save,df_GBC_save,df_feature_5_models], axis=1)


# In[216]:


df_all_save


# In[217]:


df_all_save.columns = ['Logistic_Regression', 'SVM','DecisionTree','RandomForest','XGBoost','GradientBoost','5_models_overlap']


# In[219]:


df_all_save.to_csv("./new-data-220328/RFECV_gridscores//RFECV_features_all.txt",sep="\t")


# # Select features by different methods (by RFE method)

# In[152]:


# # 6 model overlap
print("Overlap features of 6 models is: %s" % feature_6_models)

# # Different models (Num. of features)
print("Optimal N of LR features is: %d" % selector_LR.n_features_)
print("Optimal N of SVC features is: %d" % selector_SVC.n_features_)
print("Optimal N of DT features is: %d" % selector_DT.n_features_)
print("Optimal N of RF features is: %d" % selector_RF.n_features_)
print("Optimal N of XGB features is: %d" % selector_XGB.n_features_)
print("Optimal N of GBC features is: %d" % selector_GBC.n_features_)

# # Different models (AUC of features)
print("Optimal AUC of LR features is : %f" % avg_LR.max())
print("Optimal AUC of SVC features is : %f" % avg_SVC.max())
print("Optimal AUC of DT features is : %f" % avg_DT.max())
print("Optimal AUC of RF features is : %f" % avg_RF.max())
print("Optimal AUC of XGB features is : %f" % avg_XGB.max())
print("Optimal AUC of GBC features is : %f" % avg_GBC.max())


# In[153]:


# set_LR
# set_SVC
# set_DT
# set_RF
# set_XGB
# set_GBC
# feature_6_models


# # Get Ready data (After model feature selection)

# In[152]:


set_LR={'Ala','Alcohol','Asn','BMI','BW','C0', 'C10', 'C12DC', 'C141','C142OH','C182','C2','C3','C4','C41','C5','C6C41DC',
        'C8','CHOL','Cit','Creatinine','DBP','DM','DM_drug','HDL','HT','HTN_drug','Kynurenine','LDL','Lipid_drug','Lys',
        'PCaaC300','PCaaC360','PCaaC364','PCaaC365','PCaaC404','PCaaC406','PCaaC422','PCaaC424','PCaeC302','PCaeC342','PCaeC360',
        'PCaeC362','PCaeC383','PCaeC405','PCaeC424','Phe','Pro','SMC180','SMC181','SMC241','SMOHC161','SMOHC221','SMOHC222',
        'Sarcosine','Ser','Smoking','Thr','Trp','age','lysoPCaC260', 'lysoPCaC281'}


# In[153]:


feature_6_models={'Trp', 'Kynurenine', 'BMI', 'C182', 'C4', 'HTN_drug'}


# In[154]:


feature_5_models={'BW', 'Pro', 'HDL', 'PCaaC364', 'HTN_drug', 'C4', 'Ser', 'Lipid_drug', 'LDL', 'C141', 'Creatinine', 'Trp', 'C0', 'PCaaC406', 'SMOHC161', 'Sarcosine', 'C3', 'C10', 'DBP', 'C182', 'C8', 'age', 'Phe', 'BMI', 'Kynurenine', 'DM', 'CHOL'}


# In[155]:


len(set_LR),len(feature_6_models),len(feature_5_models)


# In[156]:


import pandas as pd
pd.options.mode.chained_assignment = None


# In[157]:


X_train_final=X_train[feature_5_models]
X_test_final=X_test[feature_5_models]
y_train_final=y_train
y_test_final=y_test


# In[158]:


#X_train_final


# In[159]:


X_train_final.columns.array


# # Receiver Operating Characteristic (ROC) with cross validation  (After model feature selection)

# In[160]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold


# In[161]:


from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold


# In[162]:


X_cv_final = X_train_final
y_cv_final = y_train_final


# In[163]:


X_cv_nmp_final=X_cv_final.to_numpy()
y_cv_nmp_final=y_cv_final.to_numpy()


# In[164]:


len(X_cv_nmp_final),len(y_cv_nmp_final)


# # Logistic Regression (After model feature selection)

# In[119]:


# LogisticRegression
# Classification and ROC analysis
# Run classifier with cross-validation and plot ROC curves

cv = StratifiedKFold(n_splits=10)
#classifier = svm.SVC(kernel="linear", probability=True)
classifier_LR_final = LogisticRegression(max_iter=3000)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(8, 8))

for i, (train, test) in enumerate(cv.split(X_cv_nmp_final, y_cv_nmp_final)):
    classifier_LR_final.fit(X_cv_nmp_final[train], y_cv_nmp_final[train])
    viz = RocCurveDisplay.from_estimator(
        classifier_LR_final,
        X_cv_nmp_final[test],
        y_cv_nmp_final[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic of LogisticRegression",
)
ax.set_title('Receiver operating characteristic of Logistic Regression',fontsize= 18)
ax.legend(loc="lower right")
plt.savefig("./data/LR.10fold.png",dpi=300)
plt.savefig("./data/LR.10fold.pdf")
plt.show()


# In[120]:


from sklearn.model_selection import cross_validate

# train with 10 fold corss validation
#classifier_LR_cv_final = LogisticRegression(max_iter=3000)
classifier_LR_final = LogisticRegression(max_iter=3000)
scoring=['accuracy','roc_auc','recall','precision','f1']
#scores_LR_cv_final = cross_validate(classifier_LR_cv_final,X_cv_nmp_final,y_cv_nmp_final,cv=10,scoring=scoring)
scores_LR_cv_final = cross_validate(classifier_LR_final,X_cv_nmp_final,y_cv_nmp_final,cv=10,scoring=scoring)
#print(sorted(scores_LR_cv.keys()))
#print(scores_LR_cv['test_recall_macro'])
print("10 fold Accuracy: %0.2f (± %0.2f)" % (scores_LR_cv_final['test_accuracy'].mean(),scores_LR_cv_final['test_accuracy'].std()*2))
print("10 fold AUC: %0.2f (± %0.2f)" % (scores_LR_cv_final['test_roc_auc'].mean(),scores_LR_cv_final['test_roc_auc'].std()*2))
print("10 fold Recall: %0.2f (± %0.2f)" % (scores_LR_cv_final['test_recall'].mean(),scores_LR_cv_final['test_recall'].std()*2))
print("10 fold Precision: %0.2f (± %0.2f)" % (scores_LR_cv_final['test_precision'].mean(),scores_LR_cv_final['test_precision'].std()*2))
print("10 fold f1: %0.2f (± %0.2f)" % (scores_LR_cv_final['test_f1'].mean(),scores_LR_cv_final['test_f1'].std()*2))


# In[121]:


#############################################################

## train
classifier_LR_final.fit(X_train_final, y_train_final)

## test
predicted_prob_LR_final = classifier_LR_final.predict_proba(X_test_final)[:,1]
predicted = classifier_LR_final.predict(X_test_final)

#############################################################

## Accuray e AUC
accuracy = metrics.accuracy_score(y_test_final, predicted)
auc_test = metrics.roc_auc_score(y_test_final, predicted_prob_LR_final)
print("Accuracy (overall correct predictions):",  round(accuracy,2))
print("Auc:", round(auc_test,2))
    
## Precision e Recall
recall = metrics.recall_score(y_test_final, predicted)
precision = metrics.precision_score(y_test_final, predicted)
print("Recall (all 1s predicted right):", round(recall,2))
print("Precision (confidence when predicting a 1):", round(precision,2))
print("Detail:")
print(metrics.classification_report(y_test_final, predicted, target_names=[str(i) for i in np.unique(y_test_final)]))

##############################################################

classes = np.unique(y_test_final)
fig, ax = plt.subplots(figsize=(10, 6))
cm = metrics.confusion_matrix(y_test_final, predicted, labels=classes)
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False,annot_kws={"size": 16})
ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
ax.set_title('Confusion matrix',fontsize= 18)
ax.set_yticklabels(labels=classes, rotation=0,fontsize=15)
ax.set_xticklabels(labels=classes, rotation=0,fontsize=15)
plt.show()


##############################################################


# In[122]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from numpy import argmax

# Plot the precision recall curve

classifier_LR_final.fit(X_train_final, y_train_final)
decision_scores = classifier_LR_final.decision_function(X_test_final)

precisions, recalls, thresholds = precision_recall_curve(y_test_final, decision_scores)

def plot_precision_recall_vs_thresholds(precisions, recalls, thresholds):
    plt.figure(figsize=(10,6))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    #plt.scatter(recalls[:-1],precisions[:-1], marker='o', color='black', label='Best')
    plt.xlabel("Threshold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(b=True, which="both", axis="both", color='gray', linestyle='-', linewidth=1)
    

plot_precision_recall_vs_thresholds(precisions, recalls, thresholds)
plt.show()

# convert to f score
fscore = (2 * precisions * recalls) / (precisions + recalls)
# locate the index of the largest f score
ix = argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
print("=====================================")


# Change threshold

## threshold = 0
predicted_1 = classifier_LR_final.predict(X_test_final)
print("When Threshold = 0: (defalt)")
print("When Threshold=0, the confusion matrix is:\n", confusion_matrix(y_test_final, predicted_1))
print("When Threshold=0, the precision is:",precision_score(y_test_final, predicted_1))
print("When Threshold=0, the recall is:",recall_score(y_test_final, predicted_1))
print("=====================================")

## threshold = -2.185215 (27 feature)
## threshold = -1.635653 (27 feature)
decision_score = classifier_LR_final.decision_function(X_test_final)
predicted_2 = np.array(decision_score >= -1.635653, dtype='int')
print("When Threshold = %f:" % thresholds[ix])
print("When Threshold= -1.635653, the confusion matrix is:\n",confusion_matrix(y_test_final, predicted_2))
print("When Threshold= -1.635653, the precision is:",precision_score(y_test_final, predicted_2))
print("When Threshold= -1.635653, the recall is:",recall_score(y_test_final, predicted_2))


# # SVM Classifier (After model feature selection)

# In[123]:


# LogisticRegression
# Classification and ROC analysis
# Run classifier with cross-validation and plot ROC curves

cv = StratifiedKFold(n_splits=10)
#classifier = svm.SVC(kernel="linear", probability=True)
classifier_SVC_final = SVC(kernel='rbf',probability=True , class_weight = 'balanced')
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(10, 6))

for i, (train, test) in enumerate(cv.split(X_cv_nmp_final, y_cv_nmp_final)):
    classifier_SVC_final.fit(X_cv_nmp_final[train], y_cv_nmp_final[train])
    viz = RocCurveDisplay.from_estimator(
        classifier_SVC_final,
        X_cv_nmp_final[test],
        y_cv_nmp_final[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic of SVM",
)
ax.set_title('Receiver operating characteristic of SVM',fontsize= 18)
ax.legend(loc="lower right")
plt.show()


# In[124]:


from sklearn.model_selection import cross_validate

# train with 10 fold corss validation
#classifier_SVC_cv_final = SVC(kernel='rbf',probability=True , class_weight = 'balanced')
classifier_SVC_final = SVC(kernel='rbf',probability=True , class_weight = 'balanced')
scoring=['accuracy','roc_auc','recall','precision','f1']
scores_SVC_cv_final = cross_validate(classifier_SVC_final,X_cv_nmp_final,y_cv_nmp_final,cv=10,scoring=scoring)

#print(sorted(scores_SVC_cv.keys()))
#print(scores_SVC_cv['test_recall_macro'])
print("10 fold Accuracy: %0.2f (± %0.2f)" % (scores_SVC_cv_final['test_accuracy'].mean(),scores_SVC_cv_final['test_accuracy'].std()*2))
print("10 fold AUC: %0.2f (± %0.2f)" % (scores_SVC_cv_final['test_roc_auc'].mean(),scores_SVC_cv_final['test_roc_auc'].std()*2))
print("10 fold Recall: %0.2f (± %0.2f)" % (scores_SVC_cv_final['test_recall'].mean(),scores_SVC_cv_final['test_recall'].std()*2))
print("10 fold Precision: %0.2f (± %0.2f)" % (scores_SVC_cv_final['test_precision'].mean(),scores_SVC_cv_final['test_precision'].std()*2))
print("10 fold f1: %0.2f (± %0.2f)" % (scores_SVC_cv_final['test_f1'].mean(),scores_SVC_cv_final['test_f1'].std()*2))


# In[125]:


#############################################################

## train
classifier_SVC_final.fit(X_train_final, y_train_final)
## test
predicted_prob_SVC_final = classifier_SVC_final.predict_proba(X_test_final)[:,1]
predicted = classifier_SVC_final.predict(X_test_final)

#############################################################

## Accuray e AUC
accuracy = metrics.accuracy_score(y_test_final, predicted)
auc_test = metrics.roc_auc_score(y_test_final, predicted_prob_SVC_final)
print("Accuracy (overall correct predictions):",  round(accuracy,2))
print("Auc:", round(auc_test,2))
    
## Precision e Recall
recall = metrics.recall_score(y_test_final, predicted)
precision = metrics.precision_score(y_test_final, predicted)
print("Recall (all 1s predicted right):", round(recall,2))
print("Precision (confidence when predicting a 1):", round(precision,2))
print("Detail:")
print(metrics.classification_report(y_test_final, predicted, target_names=[str(i) for i in np.unique(y_test_final)]))

##############################################################

classes = np.unique(y_test_final)
fig, ax = plt.subplots(figsize=(10, 6))
cm = metrics.confusion_matrix(y_test_final, predicted, labels=classes)
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False,annot_kws={"size": 16})
ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
ax.set_title('Confusion matrix',fontsize= 18)
ax.set_yticklabels(labels=classes, rotation=0,fontsize=15)
ax.set_xticklabels(labels=classes, rotation=0,fontsize=15)
plt.show()


##############################################################


# # Decision Tree Classifier (After model feature selection)

# In[126]:


# DecisionTreeClassifier
# Classification and ROC analysis
# Run classifier with cross-validation and plot ROC curves

cv = StratifiedKFold(n_splits=10)
#classifier = svm.SVC(kernel="linear", probability=True)
classifier_DT_final = DecisionTreeClassifier(max_depth=6)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(10, 6))

for i, (train, test) in enumerate(cv.split(X_cv_nmp_final, y_cv_nmp_final)):
    classifier_DT_final.fit(X_cv_nmp_final[train], y_cv_nmp_final[train])
    viz = RocCurveDisplay.from_estimator(
        classifier_DT_final,
        X_cv_nmp_final[test],
        y_cv_nmp_final[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic of DecisionTree",
)
ax.set_title('Receiver operating characteristic of Decision Tree',fontsize= 18)
ax.legend(loc="lower right")
plt.show()


# In[127]:


from sklearn.model_selection import cross_validate

# train with 10 fold corss validation
#classifier_DT_cv_final = DecisionTreeClassifier(max_depth=6)
classifier_DT_final = DecisionTreeClassifier(max_depth=6)
scoring=['accuracy','roc_auc','recall','precision','f1']
scores_DT_cv_final = cross_validate(classifier_DT_final,X_cv_nmp_final,y_cv_nmp_final,cv=10,scoring=scoring)

#print(sorted(scores_DT_cv.keys()))
#print(scores_DT_cv['test_recall_macro'])
print("10 fold Accuracy: %0.2f (± %0.2f)" % (scores_DT_cv_final['test_accuracy'].mean(),scores_DT_cv_final['test_accuracy'].std()*2))
print("10 fold AUC: %0.2f (± %0.2f)" % (scores_DT_cv_final['test_roc_auc'].mean(),scores_DT_cv_final['test_roc_auc'].std()*2))
print("10 fold Recall: %0.2f (± %0.2f)" % (scores_DT_cv_final['test_recall'].mean(),scores_DT_cv_final['test_recall'].std()*2))
print("10 fold Precision: %0.2f (± %0.2f)" % (scores_DT_cv_final['test_precision'].mean(),scores_DT_cv_final['test_precision'].std()*2))
print("10 fold f1: %0.2f (± %0.2f)" % (scores_DT_cv_final['test_f1'].mean(),scores_DT_cv_final['test_f1'].std()*2))


# In[128]:


#############################################################

## train
classifier_DT_final.fit(X_train_final, y_train_final)
## test
predicted_prob_DT_final = classifier_DT_final.predict_proba(X_test_final)[:,1]
predicted = classifier_DT_final.predict(X_test_final)

#############################################################

## Accuray e AUC
accuracy = metrics.accuracy_score(y_test_final, predicted)
auc_test = metrics.roc_auc_score(y_test_final, predicted_prob_DT_final)
print("Accuracy (overall correct predictions):",  round(accuracy,2))
print("Auc:", round(auc_test,2))
    
## Precision e Recall
recall = metrics.recall_score(y_test_final, predicted)
precision = metrics.precision_score(y_test_final, predicted)
print("Recall (all 1s predicted right):", round(recall,2))
print("Precision (confidence when predicting a 1):", round(precision,2))
print("Detail:")
print(metrics.classification_report(y_test_final, predicted, target_names=[str(i) for i in np.unique(y_test_final)]))

##############################################################

classes = np.unique(y_test_final)
fig, ax = plt.subplots(figsize=(10, 6))
cm = metrics.confusion_matrix(y_test_final, predicted, labels=classes)
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False,annot_kws={"size": 16})
ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
ax.set_title('Confusion matrix',fontsize= 18)
ax.set_yticklabels(labels=classes, rotation=0,fontsize=15)
ax.set_xticklabels(labels=classes, rotation=0,fontsize=15)
plt.show()


##############################################################


# In[129]:


from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree

#feature name
feature_names = X_train_final.columns.values.tolist()

#class name
class_names=['Control','LAA']

text_representation = tree.export_text(classifier_DT_final,feature_names=feature_names)
print(text_representation)

fig = plt.figure(figsize=(40,20))
_ = tree.plot_tree(classifier_DT_final, 
                   feature_names=feature_names,  
                   class_names=class_names,
                   filled=True,
                   fontsize=10)


# # Random Forest Classifier (After model feature selection)

# In[130]:


# RandomForestClassifier
# Classification and ROC analysis
# Run classifier with cross-validation and plot ROC curves

cv = StratifiedKFold(n_splits=10)
#classifier = svm.SVC(kernel="linear", probability=True)
classifier_RF_final = RandomForestClassifier()
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(10, 6))

for i, (train, test) in enumerate(cv.split(X_cv_nmp_final, y_cv_nmp_final)):
    classifier_RF_final.fit(X_cv_nmp_final[train], y_cv_nmp_final[train])
    viz = RocCurveDisplay.from_estimator(
        classifier_RF_final,
        X_cv_nmp_final[test],
        y_cv_nmp_final[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic of Random Forest",
)
ax.set_title('Receiver operating characteristic of Random Forest',fontsize= 18)
ax.legend(loc="lower right")
plt.show()


# In[131]:


from sklearn.model_selection import cross_validate

# train with 10 fold corss validation
#classifier_RF_cv_final = RandomForestClassifier()
classifier_RF_final = RandomForestClassifier()
scoring=['accuracy','roc_auc','recall','precision','f1']
scores_RF_cv_final = cross_validate(classifier_RF_final,X_cv_nmp_final,y_cv_nmp_final,cv=10,scoring=scoring)

#print(sorted(scores_RF_cv.keys()))
#print(scores_RF_cv['test_recall_macro'])
print("10 fold Accuracy: %0.2f (± %0.2f)" % (scores_RF_cv_final['test_accuracy'].mean(),scores_RF_cv_final['test_accuracy'].std()*2))
print("10 fold AUC: %0.2f (± %0.2f)" % (scores_RF_cv_final['test_roc_auc'].mean(),scores_RF_cv_final['test_roc_auc'].std()*2))
print("10 fold Recall: %0.2f (± %0.2f)" % (scores_RF_cv_final['test_recall'].mean(),scores_RF_cv_final['test_recall'].std()*2))
print("10 fold Precision: %0.2f (± %0.2f)" % (scores_RF_cv_final['test_precision'].mean(),scores_RF_cv_final['test_precision'].std()*2))
print("10 fold f1: %0.2f (± %0.2f)" % (scores_RF_cv_final['test_f1'].mean(),scores_RF_cv_final['test_f1'].std()*2))


# In[132]:


#############################################################

## train
classifier_RF_final.fit(X_train_final, y_train_final)
## test
predicted_prob_RF_final = classifier_RF_final.predict_proba(X_test_final)[:,1]
predicted = classifier_RF_final.predict(X_test_final)

#############################################################

## Accuray e AUC
accuracy = metrics.accuracy_score(y_test_final, predicted)
auc_test = metrics.roc_auc_score(y_test_final, predicted_prob_RF_final)
print("Accuracy (overall correct predictions):",  round(accuracy,2))
print("Auc:", round(auc_test,2))
    
## Precision e Recall
recall = metrics.recall_score(y_test_final, predicted)
precision = metrics.precision_score(y_test_final, predicted)
print("Recall (all 1s predicted right):", round(recall,2))
print("Precision (confidence when predicting a 1):", round(precision,2))
print("Detail:")
print(metrics.classification_report(y_test_final, predicted, target_names=[str(i) for i in np.unique(y_test_final)]))

##############################################################

classes = np.unique(y_test_final)
fig, ax = plt.subplots(figsize=(10, 6))
cm = metrics.confusion_matrix(y_test_final, predicted, labels=classes)
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False,annot_kws={"size": 16})
ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
ax.set_title('Confusion matrix',fontsize= 18)
ax.set_yticklabels(labels=classes, rotation=0,fontsize=15)
ax.set_xticklabels(labels=classes, rotation=0,fontsize=15)
plt.show()


##############################################################


# In[133]:


# Show all columns as list
feature_names = X_train_final.columns.values.tolist()
plt.figure(figsize=(10, 6))

plt.title('Feature Importance',fontsize= 15)

import time
import numpy as np

start_time = time.time()
importances = classifier_RF_final.feature_importances_
std = np.std([tree.feature_importances_ for tree in classifier_RF_final.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

import pandas as pd

forest_importances = pd.Series(importances, index=feature_names)
forest_importances.nlargest(25).plot(kind='barh')


# # XGB Classifier (After model feature selection)

# In[134]:


#list(X_train_final.columns.values)


# In[135]:


# RandomcolumnsestClassifier
# Classification and ROC analysis
# Run classifier with cross-validation and plot ROC curves

cv = StratifiedKFold(n_splits=10)
#classifier = svm.SVC(kernel="linear", probability=True)
classifier_XGB_final = XGBClassifier(objective='binary:logistic',
                          booster='gbtree',
                          eval_metric='auc',
                          tree_method='hist',
                          grow_policy='lossguide',
                          use_label_encoder=None)
tprs = []
aucs = []

fig, ax = plt.subplots(figsize=(10, 6))

for i, (train, test) in enumerate(cv.split(X_cv_nmp_final, y_cv_nmp_final)):
    classifier_XGB_final.fit(X_cv_nmp_final[train], y_cv_nmp_final[train])
    viz = RocCurveDisplay.from_estimator(
        classifier_XGB_final,
        X_cv_nmp_final[test],
        y_cv_nmp_final[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic of XGBoost",
)
ax.set_title('Receiver operating characteristic of XGBoost',fontsize= 18)
ax.legend(loc="lower right")
plt.show()


# In[136]:


from sklearn.model_selection import cross_validate

# train with 10 fold corss validation
# classifier_XGB_cv_final = XGBClassifier(objective='binary:logistic',
#                           booster='gbtree',
#                           eval_metric='auc',
#                           tree_method='hist',
#                           grow_policy='lossguide',
#                           use_label_encoder=None)

classifier_XGB_final = XGBClassifier(objective='binary:logistic',
                          booster='gbtree',
                          eval_metric='auc',
                          tree_method='hist',
                          grow_policy='lossguide',
                          use_label_encoder=None)

scoring=['accuracy','roc_auc','recall','precision','f1']

scores_XGB_cv_final = cross_validate(classifier_XGB_final,X_cv_nmp_final,y_cv_nmp_final,cv=10,scoring=scoring)

#print(sorted(scores_XGB_cv.keys()))
#print(scores_XGB_cv['test_recall_macro'])
print("10 fold Accuracy: %0.2f (± %0.2f)" % (scores_XGB_cv_final['test_accuracy'].mean(),scores_XGB_cv_final['test_accuracy'].std()*2))
print("10 fold AUC: %0.2f (± %0.2f)" % (scores_XGB_cv_final['test_roc_auc'].mean(),scores_XGB_cv_final['test_roc_auc'].std()*2))
print("10 fold Recall: %0.2f (± %0.2f)" % (scores_XGB_cv_final['test_recall'].mean(),scores_XGB_cv_final['test_recall'].std()*2))
print("10 fold Precision: %0.2f (± %0.2f)" % (scores_XGB_cv_final['test_precision'].mean(),scores_XGB_cv_final['test_precision'].std()*2))
print("10 fold f1: %0.2f (± %0.2f)" % (scores_XGB_cv_final['test_f1'].mean(),scores_XGB_cv_final['test_f1'].std()*2))


# In[137]:


#############################################################

## train
classifier_XGB_final.fit(X_train_final, y_train_final)
## test
predicted_prob_XGB_final = classifier_XGB_final.predict_proba(X_test_final)[:,1]
predicted = classifier_XGB_final.predict(X_test_final)

#############################################################

## Accuray e AUC
accuracy = metrics.accuracy_score(y_test_final, predicted)
auc_test = metrics.roc_auc_score(y_test_final, predicted_prob_XGB_final)
print("Accuracy (overall correct predictions):",  round(accuracy,2))
print("Auc:", round(auc_test,2))
    
## Precision e Recall
recall = metrics.recall_score(y_test_final, predicted)
precision = metrics.precision_score(y_test_final, predicted)
print("Recall (all 1s predicted right):", round(recall,2))
print("Precision (confidence when predicting a 1):", round(precision,2))
print("Detail:")
print(metrics.classification_report(y_test_final, predicted, target_names=[str(i) for i in np.unique(y_test_final)]))

##############################################################

classes = np.unique(y_test_final)
fig, ax = plt.subplots(figsize=(10, 6))
cm = metrics.confusion_matrix(y_test_final, predicted, labels=classes)
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False,annot_kws={"size": 16})
ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
ax.set_title('Confusion matrix',fontsize= 18)
ax.set_yticklabels(labels=classes, rotation=0,fontsize=15)
ax.set_xticklabels(labels=classes, rotation=0,fontsize=15)
plt.show()


##############################################################


# # Gradient Boost (After model feature selection)

# In[138]:


# Gradient Boost
# Classification and ROC analysis
# Run classifier with cross-validation and plot ROC curves

cv = StratifiedKFold(n_splits=10)
#classifier = svm.SVC(kernel="linear", probability=True)
classifier_GBC_final = GradientBoostingClassifier()
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(10, 6))

for i, (train, test) in enumerate(cv.split(X_cv_nmp_final, y_cv_nmp_final)):
    classifier_GBC_final.fit(X_cv_nmp_final[train], y_cv_nmp_final[train])
    viz = RocCurveDisplay.from_estimator(
        classifier_GBC_final,
        X_cv_nmp_final[test],
        y_cv_nmp_final[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic of GradientBoost",
)
ax.set_title('Receiver operating characteristic of GradientBoost',fontsize= 18)
ax.legend(loc="lower right")
plt.show()


# In[139]:


from sklearn.model_selection import cross_validate

# train with 10 fold corss validation
#classifier_GBC_cv_final = GradientBoostingClassifier()

classifier_GBC_final = GradientBoostingClassifier()

scoring=['accuracy','roc_auc','recall','precision','f1']

scores_GBC_cv_final = cross_validate(classifier_GBC_final,X_cv_nmp_final,y_cv_nmp_final,cv=10,scoring=scoring)

#print(sorted(scores_GBC_cv.keys()))
#print(scores_GBC_cv['test_recall_macro'])
print("10 fold Accuracy: %0.2f (± %0.2f)" % (scores_GBC_cv_final['test_accuracy'].mean(),scores_GBC_cv_final['test_accuracy'].std()*2))
print("10 fold AUC: %0.2f (± %0.2f)" % (scores_GBC_cv_final['test_roc_auc'].mean(),scores_GBC_cv_final['test_roc_auc'].std()*2))
print("10 fold Recall: %0.2f (± %0.2f)" % (scores_GBC_cv_final['test_recall'].mean(),scores_GBC_cv_final['test_recall'].std()*2))
print("10 fold Precision: %0.2f (± %0.2f)" % (scores_GBC_cv_final['test_precision'].mean(),scores_GBC_cv_final['test_precision'].std()*2))
print("10 fold f1: %0.2f (± %0.2f)" % (scores_GBC_cv_final['test_f1'].mean(),scores_GBC_cv_final['test_f1'].std()*2))


# In[140]:


#############################################################

## train
classifier_GBC_final.fit(X_train_final, y_train_final)

## test
predicted_prob_GBC_final = classifier_GBC_final.predict_proba(X_test_final)[:,1]
predicted = classifier_GBC_final.predict(X_test_final)

#############################################################

## Accuray e AUC
accuracy = metrics.accuracy_score(y_test_final, predicted)
auc_test = metrics.roc_auc_score(y_test_final, predicted_prob_GBC_final)
print("Accuracy (overall correct predictions):",  round(accuracy,2))
print("Auc:", round(auc_test,2))
    
## Precision e Recall
recall = metrics.recall_score(y_test_final, predicted)
precision = metrics.precision_score(y_test_final, predicted)
print("Recall (all 1s predicted right):", round(recall,2))
print("Precision (confidence when predicting a 1):", round(precision,2))
print("Detail:")
print(metrics.classification_report(y_test_final, predicted, target_names=[str(i) for i in np.unique(y_test_final)]))

##############################################################

classes = np.unique(y_test_final)
fig, ax = plt.subplots(figsize=(10, 6))
cm = metrics.confusion_matrix(y_test_final, predicted, labels=classes)
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False,annot_kws={"size": 16})
ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
ax.set_title('Confusion matrix',fontsize= 18)
ax.set_yticklabels(labels=classes, rotation=0,fontsize=15)
ax.set_xticklabels(labels=classes, rotation=0,fontsize=15)
plt.show()


##############################################################


# In[141]:


def roc_curve_and_score(y_test, pred_proba):
    fpr, tpr, _ = roc_curve(y_test.ravel(), pred_proba.ravel())
    roc_auc = roc_auc_score(y_test.ravel(), pred_proba.ravel())
    return fpr, tpr, roc_auc

plt.figure(figsize=(5, 5))
plt.rcParams.update({'font.size': 10})
plt.grid()


fpr, tpr, roc_auc = roc_curve_and_score(y_test_final, predicted_prob_LR_final)
plt.plot(fpr, tpr, color='blue', lw=2,
         label='Logistic Regression (area={0:.3f})'.format(roc_auc))
# fpr, tpr, roc_auc = roc_curve_and_score(y_test_final, predicted_prob_SVC_final)
# plt.plot(fpr, tpr, color='green', lw=2,
#          label='SVM (area={0:.3f})'.format(roc_auc))
# fpr, tpr, roc_auc = roc_curve_and_score(y_test_final, predicted_prob_DT_final)
# plt.plot(fpr, tpr, color='red', lw=2,
#          label='Decision Tree (area={0:.3f})'.format(roc_auc))
# fpr, tpr, roc_auc = roc_curve_and_score(y_test_final, predicted_prob_RF_final)
# plt.plot(fpr, tpr, color='black', lw=2,
#          label='Random Forest (area={0:.3f})'.format(roc_auc))
# fpr, tpr, roc_auc = roc_curve_and_score(y_test_final, predicted_prob_XGB_final)
# plt.plot(fpr, tpr, color='orange', lw=2,
#          label='XGBoost (area={0:.3f})'.format(roc_auc))
# fpr, tpr, roc_auc = roc_curve_and_score(y_test_final, predicted_prob_GBC_final)
# plt.plot(fpr, tpr, color='brown', lw=2,
#          label='Gradient Boost (area={0:.3f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.legend(loc="lower right")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.suptitle('ROC curves of models (With feature selection)',fontsize=15)
plt.savefig("./data/plot.LR.png", dpi=300)
plt.savefig("./data/plot.LR.pdf")
plt.show()


# # Final All features form RFE

# In[349]:


# # 6 model overlap
print("Overlap features of 5 models is: %s " % feature_5_models)

# # Different models (Num. of features)
print("Optimal N of LR features is: %d" % selector_LR.n_features_)
print("Optimal N of SVC features is: %d" % selector_SVC.n_features_)
print("Optimal N of DT features is: %d" % selector_DT.n_features_)
print("Optimal N of RF features is: %d" % selector_RF.n_features_)
print("Optimal N of XGB features is: %d" % selector_XGB.n_features_)
print("Optimal N of GBC features is: %d" % selector_GBC.n_features_)

# # Different models (AUC of features)
print("Optimal AUC of LR features is : %f" % avg_LR.max())
print("Optimal AUC of SVC features is : %f" % avg_SVC.max())
print("Optimal AUC of DT features is : %f" % avg_DT.max())
print("Optimal AUC of RF features is : %f" % avg_RF.max())
print("Optimal AUC of XGB features is : %f" % avg_XGB.max())
print("Optimal AUC of GBC features is : %f" % avg_GBC.max())


# In[ ]:




