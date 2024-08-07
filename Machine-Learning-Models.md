```python
import os
#import selenium
os.chdir("../../../LAA/")
```


```python
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
```


```python
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
```


```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
```


```python
# !python -m pip show scikit-learn
# !python -m pip freeze
```

# Load data (Imputed)


```python
laa=pd.read_csv("./new-data-220328/LAA-clinic.miRNA.0328.imputed",sep="\t", engine='python')
```


```python
#print("Dataset has {} entries and {} features".format(*laa.shape))
```


```python
laa.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Group</th>
      <th>age</th>
      <th>sex</th>
      <th>HT</th>
      <th>DM</th>
      <th>Smoking</th>
      <th>Alcohol</th>
      <th>FHx stroke</th>
      <th>CKD</th>
      <th>...</th>
      <th>SMC202</th>
      <th>SMC240</th>
      <th>SMC241</th>
      <th>SMC260</th>
      <th>SMC261</th>
      <th>miR21_mean</th>
      <th>miR155_mean</th>
      <th>miR126_mean</th>
      <th>let-7g_mean</th>
      <th>miR39_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAX637</td>
      <td>N</td>
      <td>63</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0.188</td>
      <td>11.4</td>
      <td>38.6</td>
      <td>0.145</td>
      <td>0.411</td>
      <td>28.620</td>
      <td>29.940</td>
      <td>24.555</td>
      <td>28.620</td>
      <td>26.295</td>
    </tr>
    <tr>
      <th>1</th>
      <td>304189</td>
      <td>N</td>
      <td>70</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.277</td>
      <td>22.2</td>
      <td>49.1</td>
      <td>0.163</td>
      <td>0.383</td>
      <td>29.070</td>
      <td>30.305</td>
      <td>26.910</td>
      <td>29.340</td>
      <td>25.305</td>
    </tr>
    <tr>
      <th>2</th>
      <td>300410</td>
      <td>N</td>
      <td>66</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.493</td>
      <td>24.4</td>
      <td>113.0</td>
      <td>0.220</td>
      <td>0.444</td>
      <td>29.045</td>
      <td>29.800</td>
      <td>25.600</td>
      <td>27.975</td>
      <td>26.550</td>
    </tr>
    <tr>
      <th>3</th>
      <td>303625</td>
      <td>N</td>
      <td>53</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0.266</td>
      <td>22.8</td>
      <td>53.1</td>
      <td>0.215</td>
      <td>0.632</td>
      <td>25.195</td>
      <td>28.270</td>
      <td>21.350</td>
      <td>24.530</td>
      <td>24.080</td>
    </tr>
    <tr>
      <th>4</th>
      <td>302712</td>
      <td>LAA</td>
      <td>68</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.518</td>
      <td>31.2</td>
      <td>121.0</td>
      <td>0.275</td>
      <td>0.484</td>
      <td>27.070</td>
      <td>30.605</td>
      <td>23.800</td>
      <td>28.080</td>
      <td>25.795</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 200 columns</p>
</div>




```python
laa=laa.drop(columns=['ID'])
```

# Define model columns


```python
columns_clinical = [ 'Group','age','sex','HT','DM','Smoking','Alcohol','FHx stroke','CKD','Lipid_drug','DM_drug', 'HTN_drug',
                    'BH', 'BW', 'BMI','waistline', 'Hip','SBP', 'DBP','MBP','HeartRate', 'Homocysteine', 'AcSugar', 'HsCRP', 
                    'HDL', 'LDL', 'TG', 'CHOL', 'UA', 'Creatinine']
```


```python
len(columns_clinical)
```




    30




```python
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
```


```python
len(columns_metabo)
```




    165




```python
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
```


```python
len(columns_all)
```




    194



# Define model (3 kind of models)


```python
laa_clinical=laa[columns_clinical]
laa_clinical.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Group</th>
      <th>age</th>
      <th>sex</th>
      <th>HT</th>
      <th>DM</th>
      <th>Smoking</th>
      <th>Alcohol</th>
      <th>FHx stroke</th>
      <th>CKD</th>
      <th>Lipid_drug</th>
      <th>...</th>
      <th>HeartRate</th>
      <th>Homocysteine</th>
      <th>AcSugar</th>
      <th>HsCRP</th>
      <th>HDL</th>
      <th>LDL</th>
      <th>TG</th>
      <th>CHOL</th>
      <th>UA</th>
      <th>Creatinine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>N</td>
      <td>63</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>82.0</td>
      <td>13.200000</td>
      <td>96.000000</td>
      <td>1.470000</td>
      <td>56.0</td>
      <td>104.0</td>
      <td>72.0</td>
      <td>174.0</td>
      <td>7.5</td>
      <td>1.13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>N</td>
      <td>70</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>62.0</td>
      <td>11.103058</td>
      <td>99.000000</td>
      <td>4.398697</td>
      <td>49.0</td>
      <td>127.0</td>
      <td>87.0</td>
      <td>193.0</td>
      <td>7.1</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>2</th>
      <td>N</td>
      <td>66</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>58.0</td>
      <td>14.000000</td>
      <td>92.000000</td>
      <td>0.230000</td>
      <td>40.0</td>
      <td>109.0</td>
      <td>116.0</td>
      <td>172.0</td>
      <td>4.3</td>
      <td>1.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>N</td>
      <td>53</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>77.0</td>
      <td>8.000000</td>
      <td>94.000000</td>
      <td>1.140000</td>
      <td>58.0</td>
      <td>133.0</td>
      <td>79.0</td>
      <td>207.0</td>
      <td>7.8</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LAA</td>
      <td>68</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>70.0</td>
      <td>8.800000</td>
      <td>102.759819</td>
      <td>3.010000</td>
      <td>30.0</td>
      <td>64.0</td>
      <td>291.0</td>
      <td>152.0</td>
      <td>7.5</td>
      <td>1.10</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
laa_metabo=laa[columns_metabo]
laa_metabo.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Group</th>
      <th>C0</th>
      <th>C10</th>
      <th>C101</th>
      <th>C102</th>
      <th>C12</th>
      <th>C12DC</th>
      <th>C121</th>
      <th>C14</th>
      <th>C141</th>
      <th>...</th>
      <th>SMOHC241</th>
      <th>SMC160</th>
      <th>SMC161</th>
      <th>SMC180</th>
      <th>SMC181</th>
      <th>SMC202</th>
      <th>SMC240</th>
      <th>SMC241</th>
      <th>SMC260</th>
      <th>SMC261</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>N</td>
      <td>30.9</td>
      <td>0.309</td>
      <td>0.444</td>
      <td>0.058</td>
      <td>0.085</td>
      <td>0.039</td>
      <td>0.238</td>
      <td>0.038</td>
      <td>0.062</td>
      <td>...</td>
      <td>0.556</td>
      <td>70.7</td>
      <td>10.4</td>
      <td>11.3</td>
      <td>6.55</td>
      <td>0.188</td>
      <td>11.4</td>
      <td>38.6</td>
      <td>0.145</td>
      <td>0.411</td>
    </tr>
    <tr>
      <th>1</th>
      <td>N</td>
      <td>45.5</td>
      <td>0.289</td>
      <td>0.528</td>
      <td>0.093</td>
      <td>0.117</td>
      <td>0.044</td>
      <td>0.345</td>
      <td>0.043</td>
      <td>0.083</td>
      <td>...</td>
      <td>1.040</td>
      <td>93.4</td>
      <td>13.8</td>
      <td>14.7</td>
      <td>8.30</td>
      <td>0.277</td>
      <td>22.2</td>
      <td>49.1</td>
      <td>0.163</td>
      <td>0.383</td>
    </tr>
    <tr>
      <th>2</th>
      <td>N</td>
      <td>39.6</td>
      <td>0.111</td>
      <td>0.290</td>
      <td>0.036</td>
      <td>0.073</td>
      <td>0.060</td>
      <td>0.310</td>
      <td>0.029</td>
      <td>0.043</td>
      <td>...</td>
      <td>1.340</td>
      <td>113.0</td>
      <td>15.6</td>
      <td>29.3</td>
      <td>13.40</td>
      <td>0.493</td>
      <td>24.4</td>
      <td>113.0</td>
      <td>0.220</td>
      <td>0.444</td>
    </tr>
    <tr>
      <th>3</th>
      <td>N</td>
      <td>35.9</td>
      <td>0.414</td>
      <td>0.438</td>
      <td>0.067</td>
      <td>0.133</td>
      <td>0.042</td>
      <td>0.257</td>
      <td>0.050</td>
      <td>0.094</td>
      <td>...</td>
      <td>1.240</td>
      <td>82.2</td>
      <td>11.5</td>
      <td>14.5</td>
      <td>7.75</td>
      <td>0.266</td>
      <td>22.8</td>
      <td>53.1</td>
      <td>0.215</td>
      <td>0.632</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LAA</td>
      <td>43.8</td>
      <td>0.179</td>
      <td>0.343</td>
      <td>0.060</td>
      <td>0.104</td>
      <td>0.053</td>
      <td>0.301</td>
      <td>0.030</td>
      <td>0.060</td>
      <td>...</td>
      <td>1.220</td>
      <td>110.0</td>
      <td>17.2</td>
      <td>32.2</td>
      <td>14.00</td>
      <td>0.518</td>
      <td>31.2</td>
      <td>121.0</td>
      <td>0.275</td>
      <td>0.484</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 165 columns</p>
</div>




```python
laa_all=laa[columns_all]
laa_all.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Group</th>
      <th>age</th>
      <th>sex</th>
      <th>HT</th>
      <th>DM</th>
      <th>Smoking</th>
      <th>Alcohol</th>
      <th>FHx stroke</th>
      <th>CKD</th>
      <th>BH</th>
      <th>...</th>
      <th>SMOHC241</th>
      <th>SMC160</th>
      <th>SMC161</th>
      <th>SMC180</th>
      <th>SMC181</th>
      <th>SMC202</th>
      <th>SMC240</th>
      <th>SMC241</th>
      <th>SMC260</th>
      <th>SMC261</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>N</td>
      <td>63</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>168.0</td>
      <td>...</td>
      <td>0.556</td>
      <td>70.7</td>
      <td>10.4</td>
      <td>11.3</td>
      <td>6.55</td>
      <td>0.188</td>
      <td>11.4</td>
      <td>38.6</td>
      <td>0.145</td>
      <td>0.411</td>
    </tr>
    <tr>
      <th>1</th>
      <td>N</td>
      <td>70</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>166.0</td>
      <td>...</td>
      <td>1.040</td>
      <td>93.4</td>
      <td>13.8</td>
      <td>14.7</td>
      <td>8.30</td>
      <td>0.277</td>
      <td>22.2</td>
      <td>49.1</td>
      <td>0.163</td>
      <td>0.383</td>
    </tr>
    <tr>
      <th>2</th>
      <td>N</td>
      <td>66</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>174.0</td>
      <td>...</td>
      <td>1.340</td>
      <td>113.0</td>
      <td>15.6</td>
      <td>29.3</td>
      <td>13.40</td>
      <td>0.493</td>
      <td>24.4</td>
      <td>113.0</td>
      <td>0.220</td>
      <td>0.444</td>
    </tr>
    <tr>
      <th>3</th>
      <td>N</td>
      <td>53</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>166.0</td>
      <td>...</td>
      <td>1.240</td>
      <td>82.2</td>
      <td>11.5</td>
      <td>14.5</td>
      <td>7.75</td>
      <td>0.266</td>
      <td>22.8</td>
      <td>53.1</td>
      <td>0.215</td>
      <td>0.632</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LAA</td>
      <td>68</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>165.0</td>
      <td>...</td>
      <td>1.220</td>
      <td>110.0</td>
      <td>17.2</td>
      <td>32.2</td>
      <td>14.00</td>
      <td>0.518</td>
      <td>31.2</td>
      <td>121.0</td>
      <td>0.275</td>
      <td>0.484</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 194 columns</p>
</div>



# Get Ready data (After model feature selection)

## Select 3 kind of Model


```python
# laa_clean=laa_clinical
# laa_clean=laa_metabo
laa_clean=laa_all
```


```python
#Recode factor
laa_clean['Group'] = np.where(laa_clean['Group']== "LAA", 1, 0)
laa_clean.columns
```




    Index(['Group', 'age', 'sex', 'HT', 'DM', 'Smoking', 'Alcohol', 'FHx stroke',
           'CKD', 'BH',
           ...
           'SMOHC241', 'SMC160', 'SMC161', 'SMC180', 'SMC181', 'SMC202', 'SMC240',
           'SMC241', 'SMC260', 'SMC261'],
          dtype='object', length=194)




```python
# All model
x = laa_clean.loc[:,'age':'SMC261']
y = laa_clean.loc[:,'Group']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state =2018, shuffle = True)
```


```python
X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>HT</th>
      <th>DM</th>
      <th>Smoking</th>
      <th>Alcohol</th>
      <th>FHx stroke</th>
      <th>CKD</th>
      <th>BH</th>
      <th>BW</th>
      <th>...</th>
      <th>SMOHC241</th>
      <th>SMC160</th>
      <th>SMC161</th>
      <th>SMC180</th>
      <th>SMC181</th>
      <th>SMC202</th>
      <th>SMC240</th>
      <th>SMC241</th>
      <th>SMC260</th>
      <th>SMC261</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>61</th>
      <td>53</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>167.500000</td>
      <td>83.000000</td>
      <td>...</td>
      <td>1.110</td>
      <td>125.0</td>
      <td>17.90</td>
      <td>29.20</td>
      <td>12.60</td>
      <td>0.506</td>
      <td>37.4</td>
      <td>134.0</td>
      <td>0.229</td>
      <td>0.369</td>
    </tr>
    <tr>
      <th>17</th>
      <td>60</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>158.000000</td>
      <td>64.000000</td>
      <td>...</td>
      <td>1.060</td>
      <td>127.0</td>
      <td>14.40</td>
      <td>19.90</td>
      <td>9.86</td>
      <td>0.415</td>
      <td>20.7</td>
      <td>54.1</td>
      <td>0.184</td>
      <td>0.405</td>
    </tr>
    <tr>
      <th>35</th>
      <td>50</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>163.647851</td>
      <td>67.114327</td>
      <td>...</td>
      <td>0.805</td>
      <td>82.4</td>
      <td>11.90</td>
      <td>9.05</td>
      <td>5.63</td>
      <td>0.278</td>
      <td>14.4</td>
      <td>41.0</td>
      <td>0.143</td>
      <td>0.327</td>
    </tr>
    <tr>
      <th>58</th>
      <td>64</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>168.000000</td>
      <td>67.000000</td>
      <td>...</td>
      <td>0.717</td>
      <td>74.1</td>
      <td>8.39</td>
      <td>10.40</td>
      <td>5.12</td>
      <td>0.225</td>
      <td>12.4</td>
      <td>31.0</td>
      <td>0.102</td>
      <td>0.234</td>
    </tr>
    <tr>
      <th>270</th>
      <td>61</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>155.000000</td>
      <td>56.000000</td>
      <td>...</td>
      <td>1.320</td>
      <td>159.0</td>
      <td>19.50</td>
      <td>23.10</td>
      <td>12.40</td>
      <td>0.348</td>
      <td>30.8</td>
      <td>68.1</td>
      <td>0.196</td>
      <td>0.438</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 193 columns</p>
</div>



# Standard Scalar


```python
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
```

## Select the model features 


```python
# col_names=std_sc_clinical
# col_names=std_sc_metabo
col_names=std_sc_all
```


```python
scalar=StandardScaler()
```


```python
# Train data Standard scaler
features_train = X_train[col_names]
scaler_train = scalar.fit_transform(features_train)
X_train[col_names] = scaler_train
```


```python
# Test data Standard scaler
features_test = X_test[col_names]
scaler_test = scalar.transform(features_test)
X_test[col_names] = scaler_test
```


```python
X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>HT</th>
      <th>DM</th>
      <th>Smoking</th>
      <th>Alcohol</th>
      <th>FHx stroke</th>
      <th>CKD</th>
      <th>BH</th>
      <th>BW</th>
      <th>...</th>
      <th>SMOHC241</th>
      <th>SMC160</th>
      <th>SMC161</th>
      <th>SMC180</th>
      <th>SMC181</th>
      <th>SMC202</th>
      <th>SMC240</th>
      <th>SMC241</th>
      <th>SMC260</th>
      <th>SMC261</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>61</th>
      <td>-1.329102</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.596792</td>
      <td>1.721345</td>
      <td>...</td>
      <td>-0.542633</td>
      <td>0.106593</td>
      <td>0.133885</td>
      <td>0.495498</td>
      <td>0.028754</td>
      <td>0.345622</td>
      <td>0.887814</td>
      <td>0.638801</td>
      <td>-0.098863</td>
      <td>-0.770216</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-0.388240</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.818724</td>
      <td>-0.280674</td>
      <td>...</td>
      <td>-0.665082</td>
      <td>0.166939</td>
      <td>-0.571051</td>
      <td>-0.570159</td>
      <td>-0.637670</td>
      <td>-0.257317</td>
      <td>-0.786580</td>
      <td>-0.947036</td>
      <td>-0.664574</td>
      <td>-0.600971</td>
    </tr>
    <tr>
      <th>35</th>
      <td>-1.732328</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.022816</td>
      <td>0.047481</td>
      <td>...</td>
      <td>-1.289573</td>
      <td>-1.178769</td>
      <td>-1.074576</td>
      <td>-1.813425</td>
      <td>-1.666494</td>
      <td>-1.165039</td>
      <td>-1.418237</td>
      <td>-1.207042</td>
      <td>-1.180000</td>
      <td>-0.967667</td>
    </tr>
    <tr>
      <th>58</th>
      <td>0.149395</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.671293</td>
      <td>0.035434</td>
      <td>...</td>
      <td>-1.505084</td>
      <td>-1.429203</td>
      <td>-1.781526</td>
      <td>-1.658733</td>
      <td>-1.790536</td>
      <td>-1.516201</td>
      <td>-1.618763</td>
      <td>-1.405520</td>
      <td>-1.695427</td>
      <td>-1.404882</td>
    </tr>
    <tr>
      <th>270</th>
      <td>-0.253831</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-1.265729</td>
      <td>-1.123629</td>
      <td>...</td>
      <td>-0.028347</td>
      <td>1.132469</td>
      <td>0.456142</td>
      <td>-0.203481</td>
      <td>-0.019890</td>
      <td>-0.701239</td>
      <td>0.226078</td>
      <td>-0.669168</td>
      <td>-0.513718</td>
      <td>-0.445831</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 193 columns</p>
</div>




```python
X_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>HT</th>
      <th>DM</th>
      <th>Smoking</th>
      <th>Alcohol</th>
      <th>FHx stroke</th>
      <th>CKD</th>
      <th>BH</th>
      <th>BW</th>
      <th>...</th>
      <th>SMOHC241</th>
      <th>SMC160</th>
      <th>SMC161</th>
      <th>SMC180</th>
      <th>SMC181</th>
      <th>SMC202</th>
      <th>SMC240</th>
      <th>SMC241</th>
      <th>SMC260</th>
      <th>SMC261</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>157</th>
      <td>0.014986</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.341801</td>
      <td>1.405236</td>
      <td>...</td>
      <td>-0.420184</td>
      <td>-0.931352</td>
      <td>-1.316268</td>
      <td>-0.444114</td>
      <td>-0.968450</td>
      <td>-0.687988</td>
      <td>-1.007158</td>
      <td>-0.036023</td>
      <td>-0.186862</td>
      <td>-0.300093</td>
    </tr>
    <tr>
      <th>136</th>
      <td>-1.463511</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.267300</td>
      <td>1.289330</td>
      <td>...</td>
      <td>0.290021</td>
      <td>-0.014098</td>
      <td>-0.007102</td>
      <td>0.518415</td>
      <td>0.539518</td>
      <td>0.166728</td>
      <td>0.366446</td>
      <td>0.976213</td>
      <td>0.378849</td>
      <td>0.132421</td>
    </tr>
    <tr>
      <th>293</th>
      <td>0.821439</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-1.191228</td>
      <td>-2.071954</td>
      <td>...</td>
      <td>2.298188</td>
      <td>2.520419</td>
      <td>1.966718</td>
      <td>0.380911</td>
      <td>1.196213</td>
      <td>1.193711</td>
      <td>1.048235</td>
      <td>-0.182897</td>
      <td>1.019989</td>
      <td>1.006850</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.388240</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.565304</td>
      <td>1.089128</td>
      <td>...</td>
      <td>-1.446308</td>
      <td>-1.703776</td>
      <td>-1.622412</td>
      <td>-1.452477</td>
      <td>-1.423273</td>
      <td>-1.317430</td>
      <td>-1.358079</td>
      <td>-1.310251</td>
      <td>-1.632570</td>
      <td>-1.188625</td>
    </tr>
    <tr>
      <th>36</th>
      <td>-0.925875</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.520721</td>
      <td>2.037453</td>
      <td>...</td>
      <td>-0.101816</td>
      <td>-1.012819</td>
      <td>-1.215563</td>
      <td>-0.192023</td>
      <td>-0.968450</td>
      <td>-0.429586</td>
      <td>-0.225106</td>
      <td>0.043368</td>
      <td>-0.287433</td>
      <td>-0.163757</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 193 columns</p>
</div>




```python
print(len(X_train),len(X_test))
```

    287 72


# Receiver Operating Characteristic (ROC) with cross validation


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
```


```python
from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
```


```python
X_cv = X_train
y_cv = y_train
```


```python
X_cv_nmp=X_cv.to_numpy()
y_cv_nmp=y_cv.to_numpy()
```

# Logistic Regression


```python
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
```


    
![png](output_43_0.png)
    



```python
len(X_cv_nmp),len(y_cv_nmp)
```




    (287, 287)




```python
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
```

    10 fold Accuracy: 0.78 (± 0.15)
    10 fold AUC: 0.89 (± 0.12)
    10 fold Recall: 0.77 (± 0.26)
    10 fold Precision: 0.77 (± 0.17)
    10 fold f1: 0.76 (± 0.18)



```python
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
```

    Accuracy (overall correct predictions): 0.83
    Auc: 0.9
    Recall (all 1s predicted right): 0.72
    Precision (confidence when predicting a 1): 0.97
    Detail:
                  precision    recall  f1-score   support
    
               0       0.74      0.97      0.84        32
               1       0.97      0.72      0.83        40
    
        accuracy                           0.83        72
       macro avg       0.85      0.85      0.83        72
    weighted avg       0.87      0.83      0.83        72
    



    
![png](output_46_1.png)
    


# SVM Classifier


```python
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
```


    
![png](output_48_0.png)
    



```python
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
```

    10 fold Accuracy: 0.77 (± 0.16)
    10 fold AUC: 0.85 (± 0.13)
    10 fold Recall: 0.77 (± 0.20)
    10 fold Precision: 0.76 (± 0.22)
    10 fold f1: 0.76 (± 0.17)



```python
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
```

    Accuracy (overall correct predictions): 0.78
    Auc: 0.89
    Recall (all 1s predicted right): 0.72
    Precision (confidence when predicting a 1): 0.85
    Detail:
                  precision    recall  f1-score   support
    
               0       0.71      0.84      0.77        32
               1       0.85      0.72      0.78        40
    
        accuracy                           0.78        72
       macro avg       0.78      0.78      0.78        72
    weighted avg       0.79      0.78      0.78        72
    



    
![png](output_50_1.png)
    


# Decision Tree Classifier


```python
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
```


    
![png](output_52_0.png)
    



```python
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
```

    10 fold Accuracy: 0.72 (± 0.11)
    10 fold AUC: 0.71 (± 0.12)
    10 fold Recall: 0.68 (± 0.22)
    10 fold Precision: 0.72 (± 0.11)
    10 fold f1: 0.69 (± 0.13)



```python
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
```

    Accuracy (overall correct predictions): 0.68
    Auc: 0.67
    Recall (all 1s predicted right): 0.65
    Precision (confidence when predicting a 1): 0.74
    Detail:
                  precision    recall  f1-score   support
    
               0       0.62      0.72      0.67        32
               1       0.74      0.65      0.69        40
    
        accuracy                           0.68        72
       macro avg       0.68      0.68      0.68        72
    weighted avg       0.69      0.68      0.68        72
    



    
![png](output_54_1.png)
    



```python
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
```

    |--- HTN_drug <= 0.50
    |   |--- Smoking <= 0.50
    |   |   |--- Kynurenine <= 1.22
    |   |   |   |--- C12 <= -0.97
    |   |   |   |   |--- PCaaC320 <= -0.36
    |   |   |   |   |   |--- class: 0
    |   |   |   |   |--- PCaaC320 >  -0.36
    |   |   |   |   |   |--- class: 1
    |   |   |   |--- C12 >  -0.97
    |   |   |   |   |--- age <= 0.89
    |   |   |   |   |   |--- class: 0
    |   |   |   |   |--- age >  0.89
    |   |   |   |   |   |--- Tyr <= -0.05
    |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- Tyr >  -0.05
    |   |   |   |   |   |   |--- class: 0
    |   |   |--- Kynurenine >  1.22
    |   |   |   |--- class: 1
    |   |--- Smoking >  0.50
    |   |   |--- Trp <= -0.26
    |   |   |   |--- Val <= -1.13
    |   |   |   |   |--- Tyr <= -1.32
    |   |   |   |   |   |--- class: 1
    |   |   |   |   |--- Tyr >  -1.32
    |   |   |   |   |   |--- class: 0
    |   |   |   |--- Val >  -1.13
    |   |   |   |   |--- Thr <= 0.53
    |   |   |   |   |   |--- C4 <= 2.03
    |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- C4 >  2.03
    |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |--- Thr >  0.53
    |   |   |   |   |   |--- class: 0
    |   |   |--- Trp >  -0.26
    |   |   |   |--- Arg <= 1.66
    |   |   |   |   |--- C2 <= -1.01
    |   |   |   |   |   |--- class: 1
    |   |   |   |   |--- C2 >  -1.01
    |   |   |   |   |   |--- lysoPCaC181 <= 0.26
    |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |--- lysoPCaC181 >  0.26
    |   |   |   |   |   |   |--- class: 0
    |   |   |   |--- Arg >  1.66
    |   |   |   |   |--- class: 1
    |--- HTN_drug >  0.50
    |   |--- C4 <= -0.62
    |   |   |--- PCaeC361 <= -1.07
    |   |   |   |--- class: 1
    |   |   |--- PCaeC361 >  -1.07
    |   |   |   |--- C182 <= -0.24
    |   |   |   |   |--- class: 0
    |   |   |   |--- C182 >  -0.24
    |   |   |   |   |--- class: 1
    |   |--- C4 >  -0.62
    |   |   |--- Kynurenine <= -1.05
    |   |   |   |--- Creatinine <= -0.22
    |   |   |   |   |--- class: 0
    |   |   |   |--- Creatinine >  -0.22
    |   |   |   |   |--- class: 1
    |   |   |--- Kynurenine >  -1.05
    |   |   |   |--- PCaeC380 <= 2.21
    |   |   |   |   |--- BMI <= 1.60
    |   |   |   |   |   |--- BH <= 2.03
    |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- BH >  2.03
    |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |--- BMI >  1.60
    |   |   |   |   |   |--- C5DCC6OH <= -0.20
    |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |--- C5DCC6OH >  -0.20
    |   |   |   |   |   |   |--- class: 1
    |   |   |   |--- PCaeC380 >  2.21
    |   |   |   |   |--- class: 0
    



    
![png](output_55_1.png)
    


# Random Forest Classifier


```python
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
```


    
![png](output_57_0.png)
    



```python
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
```

    10 fold Accuracy: 0.76 (± 0.17)
    10 fold AUC: 0.87 (± 0.10)
    10 fold Recall: 0.71 (± 0.26)
    10 fold Precision: 0.76 (± 0.18)
    10 fold f1: 0.73 (± 0.20)



```python
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
```

    Accuracy (overall correct predictions): 0.83
    Auc: 0.9
    Recall (all 1s predicted right): 0.75
    Precision (confidence when predicting a 1): 0.94
    Detail:
                  precision    recall  f1-score   support
    
               0       0.75      0.94      0.83        32
               1       0.94      0.75      0.83        40
    
        accuracy                           0.83        72
       macro avg       0.84      0.84      0.83        72
    weighted avg       0.85      0.83      0.83        72
    



    
![png](output_59_1.png)
    



```python
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
```

    Elapsed time to compute the importances: 0.011 seconds





    <AxesSubplot:title={'center':'Feature Importance'}>




    
![png](output_60_2.png)
    


# XGB Classifier


```python
#list(X_train.columns.values)
```


```python
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
```


    
![png](output_63_0.png)
    



```python
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
```

    10 fold Accuracy: 0.80 (± 0.11)
    10 fold AUC: 0.87 (± 0.11)
    10 fold Recall: 0.75 (± 0.26)
    10 fold Precision: 0.81 (± 0.12)
    10 fold f1: 0.77 (± 0.15)



```python
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
```

    Accuracy (overall correct predictions): 0.82
    Auc: 0.92
    Recall (all 1s predicted right): 0.75
    Precision (confidence when predicting a 1): 0.91
    Detail:
                  precision    recall  f1-score   support
    
               0       0.74      0.91      0.82        32
               1       0.91      0.75      0.82        40
    
        accuracy                           0.82        72
       macro avg       0.83      0.83      0.82        72
    weighted avg       0.84      0.82      0.82        72
    



    
![png](output_65_1.png)
    


# Gradient Boost


```python
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
```


    
![png](output_67_0.png)
    



```python
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
```

    10 fold Accuracy: 0.82 (± 0.13)
    10 fold AUC: 0.88 (± 0.11)
    10 fold Recall: 0.77 (± 0.22)
    10 fold Precision: 0.83 (± 0.13)
    10 fold f1: 0.80 (± 0.17)



```python
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
```

    Accuracy (overall correct predictions): 0.85
    Auc: 0.93
    Recall (all 1s predicted right): 0.75
    Precision (confidence when predicting a 1): 0.97
    Detail:
                  precision    recall  f1-score   support
    
               0       0.76      0.97      0.85        32
               1       0.97      0.75      0.85        40
    
        accuracy                           0.85        72
       macro avg       0.86      0.86      0.85        72
    weighted avg       0.87      0.85      0.85        72
    



    
![png](output_69_1.png)
    



```python
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
```


    
![png](output_70_0.png)
    


# RFE (Recursive Feature Elimination) 


```python
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
```


```python
X_train_rfe=X_train
X_test_rfe=X_test
y_train_rfe=y_train
y_test_rfe=y_test
```


```python
cv_rfe = StratifiedKFold(n_splits=10)
```

## RFE (Logistic Regression)


```python
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
```

    Optimal AUC of features is : 0.901003
    Optimal number of features in LR is: 62



```python
g_scores_LR
```




    array([[0.71394231, 0.53571429, 0.68333333, ..., 0.63589744, 0.62051282,
            0.71282051],
           [0.80528846, 0.75952381, 0.68571429, ..., 0.65384615, 0.67948718,
            0.77179487],
           [0.83653846, 0.82380952, 0.73809524, ..., 0.72051282, 0.76153846,
            0.80769231],
           ...,
           [0.76923077, 0.82380952, 0.95714286, ..., 0.88205128, 0.89230769,
            0.94358974],
           [0.76923077, 0.82380952, 0.95714286, ..., 0.88205128, 0.89230769,
            0.94358974],
           [0.76923077, 0.82380952, 0.95714286, ..., 0.88205128, 0.89230769,
            0.94358974]])




```python
kept_LR
```




    <PandasArray>
    [        'age',          'HT',          'DM',     'Smoking',     'Alcohol',
              'BW',         'BMI',         'DBP',         'HDL',         'LDL',
            'CHOL',  'Creatinine',  'Lipid_drug',     'DM_drug',    'HTN_drug',
              'C0',         'C10',       'C12DC',        'C141',      'C142OH',
            'C182',          'C2',          'C3',          'C4',         'C41',
         'C6C41DC',          'C5',          'C8',         'Ala',         'Asn',
             'Cit',         'Lys',         'Phe',         'Pro',         'Ser',
             'Thr',         'Trp',  'Kynurenine',   'Sarcosine', 'lysoPCaC260',
     'lysoPCaC281',    'PCaaC300',    'PCaaC360',    'PCaaC364',    'PCaaC365',
        'PCaaC404',    'PCaaC406',    'PCaaC422',    'PCaaC424',    'PCaeC302',
        'PCaeC342',    'PCaeC360',    'PCaeC362',    'PCaeC383',    'PCaeC405',
        'PCaeC424',    'SMOHC161',    'SMOHC221',    'SMOHC222',      'SMC180',
          'SMC181',      'SMC241']
    Length: 62, dtype: object




```python
g_scores_LR_df=pd.DataFrame(g_scores_LR)
g_scores_LR_df.to_csv("./new-data-220328/gridscores/g_scores_LR.txt",index=False, sep="\t")
```


```python
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
```


    
![png](output_80_0.png)
    


## RFE (SVM)


```python
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
```

    Optimal AUC of features is : 0.883022
    Optimal number of features in SVC is: 83



```python
kept_SVC
```




    <PandasArray>
    [        'age',          'HT',          'DM',     'Smoking',     'Alcohol',
              'BW',         'BMI',         'DBP',         'HDL',         'LDL',
              'TG',        'CHOL',  'Creatinine',  'Lipid_drug',     'DM_drug',
        'HTN_drug',          'C0',         'C10',        'C102',       'C12DC',
             'C14',        'C141',        'C142',      'C142OH',      'C161OH',
          'C181OH',        'C182',          'C2',          'C3',          'C4',
             'C41',     'C6C41DC',          'C5',        'C7DC',          'C8',
              'C9',         'Ala',         'Asn',         'Lys',         'Orn',
             'Phe',         'Pro',         'Ser',         'Thr',         'Trp',
      'Kynurenine',   'Sarcosine', 'lysoPCaC170', 'lysoPCaC204', 'lysoPCaC240',
     'lysoPCaC281',    'PCaaC300',    'PCaaC343',    'PCaaC344',    'PCaaC360',
        'PCaaC361',    'PCaaC362',    'PCaaC363',    'PCaaC364',    'PCaaC365',
        'PCaaC403',    'PCaaC404',    'PCaaC406',    'PCaaC422',    'PCaaC424',
        'PCaeC302',    'PCaeC322',    'PCaeC341',    'PCaeC360',    'PCaeC364',
        'PCaeC380',    'PCaeC382',    'PCaeC383',    'PCaeC401',    'PCaeC405',
        'PCaeC424',    'PCaeC444',    'SMOHC161',    'SMOHC221',    'SMOHC222',
          'SMC161',      'SMC180',      'SMC261']
    Length: 83, dtype: object




```python
g_scores_SVC_df=pd.DataFrame(g_scores_SVC)
g_scores_SVC_df.to_csv("./new-data-220328/gridscores/g_scores_SVC.txt",index=False, sep="\t")
```


```python
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
```


    
![png](output_85_0.png)
    


## RFE (Decision Tree)


```python
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
```

    Optimal AUC of features is : 0.726200
    Optimal number of features in DT is: 183



```python
kept_DT
```




    <PandasArray>
    [   'Smoking',    'Alcohol', 'FHx stroke',        'CKD',         'BH',
             'BW',        'BMI',  'waistline',        'Hip',        'SBP',
     ...
       'SMOHC222',   'SMOHC241',     'SMC160',     'SMC161',     'SMC180',
         'SMC181',     'SMC202',     'SMC240',     'SMC241',     'SMC260']
    Length: 183, dtype: object




```python
g_scores_DT_df=pd.DataFrame(g_scores_DT)
g_scores_DT_df.to_csv("./new-data-220328/gridscores/g_scores_DT.txt",index=False, sep="\t")
```


```python
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
```


    
![png](output_90_0.png)
    


## RFE (Random Forest)


```python
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
```

    Optimal AUC of features is : 0.891653
    Optimal number of features in RF is: 183



```python
kept_RF
```




    <PandasArray>
    [      'age',        'HT',        'DM',   'Smoking',        'BH',        'BW',
           'BMI', 'waistline',       'Hip',       'SBP',
     ...
      'SMOHC241',    'SMC160',    'SMC161',    'SMC180',    'SMC181',    'SMC202',
        'SMC240',    'SMC241',    'SMC260',    'SMC261']
    Length: 183, dtype: object




```python
g_scores_RF_df=pd.DataFrame(g_scores_RF)
g_scores_RF_df.to_csv("./new-data-220328/gridscores/g_scores_RF.txt",index=False, sep="\t")
```


```python
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
```


    
![png](output_95_0.png)
    


## RFE (XGB)


```python
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
```

    Optimal AUC of features is : 0.871323
    Optimal number of features in XGB is: 115



```python
kept_XGB
```




    <PandasArray>
    [     'age',       'HT',       'DM',  'Smoking',  'Alcohol',       'BW',
          'BMI',      'Hip',      'SBP',      'DBP',
     ...
     'SMOHC141', 'SMOHC161', 'SMOHC241',   'SMC180',   'SMC181',   'SMC202',
       'SMC240',   'SMC241',   'SMC260',   'SMC261']
    Length: 115, dtype: object




```python
g_scores_XGB_df=pd.DataFrame(g_scores_XGB)
g_scores_XGB_df.to_csv("./new-data-220328/gridscores/g_scores_XGB.txt",index=False, sep="\t")
```


```python
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
```


    
![png](output_100_0.png)
    


## RFE (GradientBoosting)


```python
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
```

    Optimal AUC of features is : 0.889249
    Optimal number of features in GBC is: 156



```python
kept_GBC
```




    <PandasArray>
    [       'age',        'sex',         'HT',         'DM',    'Smoking',
        'Alcohol', 'FHx stroke',        'CKD',         'BH',         'BW',
     ...
       'SMOHC241',     'SMC160',     'SMC161',     'SMC180',     'SMC181',
         'SMC202',     'SMC240',     'SMC241',     'SMC260',     'SMC261']
    Length: 156, dtype: object




```python
g_scores_GBC_df=pd.DataFrame(g_scores_GBC)
g_scores_GBC_df.to_csv("./new-data-220328/gridscores/g_scores_GBC.txt",index=False, sep="\t")
```


```python
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
```


    
![png](output_105_0.png)
    



```python
from venn import venn
from venn import pseudovenn

%matplotlib inline

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
```




    <AxesSubplot:>




    
![png](output_106_1.png)
    



```python
feature_6_models=list(set(set_LR) & set(set_SVC) & set(set_DT) & set(set_RF) & set(set_XGB) & set(set_GBC))
```


```python
len(feature_6_models)
```




    34




```python
feature_6_models
```




    ['BW',
     'Pro',
     'HDL',
     'Asn',
     'PCaaC364',
     'HTN_drug',
     'C2',
     'C4',
     'Ser',
     'SMC180',
     'Lipid_drug',
     'LDL',
     'C141',
     'Creatinine',
     'Trp',
     'C0',
     'Sarcosine',
     'PCaaC406',
     'PCaeC360',
     'C3',
     'SMOHC161',
     'C10',
     'PCaaC300',
     'Ala',
     'PCaaC360',
     'PCaaC404',
     'DBP',
     'Smoking',
     'C182',
     'C8',
     'Phe',
     'BMI',
     'Kynurenine',
     'CHOL']



# Save the model feature parameter (Features on Paper)


```python
set_LR_save={'Ala','Alcohol','Asn','BMI','BW', 'C0', 'C10', 'C12DC', 'C141', 'C142OH', 'C182', 'C2', 'C3', 'C4', 'C41', 'C5','C6C41DC', 'C8', 'CHOL', 'Cit', 'Creatinine', 'DBP', 'DM', 'DM_drug', 'HDL', 'HT', 'HTN_drug', 'Kynurenine', 'LDL', 'Lipid_drug', 'Lys', 'PCaaC300', 'PCaaC360', 'PCaaC364', 'PCaaC365', 'PCaaC404', 'PCaaC406', 'PCaaC422', 'PCaaC424', 'PCaeC302','PCaeC342','PCaeC360', 'PCaeC362','PCaeC383','PCaeC405','PCaeC424','Phe','Pro','SMC180','SMC181','SMC241','SMOHC161','SMOHC221','SMOHC222','Sarcosine','Ser','Smoking','Thr','Trp','age','lysoPCaC260','lysoPCaC281'}
len(set_LR_save)
```




    62




```python
set_SVC_save={'Ala', 'Alcohol', 'Asn', 'BMI', 'BW', 'C0', 'C10', 'C102', 'C12DC', 'C14', 'C141', 'C142', 'C142OH', 'C161OH', 'C181OH', 'C182', 'C2', 'C3', 'C4', 'C41', 'C5', 'C6C41DC', 'C7DC', 'C8', 'C9', 'CHOL', 'Creatinine', 'DBP', 'DM', 'DM_drug', 'HDL', 'HT', 'HTN_drug', 'Kynurenine', 'LDL', 'Lipid_drug', 'Lys', 'Orn', 'PCaaC300', 'PCaaC343', 'PCaaC344', 'PCaaC360', 'PCaaC361', 'PCaaC362', 'PCaaC363', 'PCaaC364', 'PCaaC365', 'PCaaC403', 'PCaaC404', 'PCaaC406', 'PCaaC422', 'PCaaC424', 'PCaeC302', 'PCaeC322', 'PCaeC341','PCaeC360', 'PCaeC364', 'PCaeC380', 'PCaeC382', 'PCaeC383', 'PCaeC401', 'PCaeC405', 'PCaeC424', 'PCaeC444', 'Phe', 'Pro', 'SMC161','SMC180', 'SMC261', 'SMOHC161', 'SMOHC221','SMOHC222', 'Sarcosine', 'Ser', 'Smoking', 'TG', 'Thr', 'Trp', 'age', 'lysoPCaC170', 'lysoPCaC204', 'lysoPCaC240', 'lysoPCaC281'}
len(set_SVC_save)
```




    83




```python
set_DT_save={'Arg', 'BMI', 'C182', 'C2', 'C4', 'C9', 'HTN_drug', 'HsCRP', 'Kynurenine', 'PCaeC361', 'Smoking', 'Trp', 'Val'}
len(set_DT_save)
```




    13




```python
set_RF_save={'AcSugar', 'Asp', 'BMI', 'BW', 'C0', 'C10','C101', 'C12', 'C121', 'C141', 'C161', 'C182', 'C3', 'C4', 'C5', 'C7DC', 'C8', 'C9', 'CHOL', 'Cit', 'Creatinine', 'Creatinine_MS', 'DBP', 'DM', 'DM_drug', 'Glu', 'HDL', 'HTN_drug', 'Hip', 'Homocysteine', 'HsCRP', 'Ile', 'Kynurenine', 'LDL', 'Leu', 'Lipid_drug', 'Lys', 'MBP', 'Met', 'PCaaC240', 'PCaaC281', 'PCaaC323', 'PCaaC343', 'PCaaC344', 'PCaaC361', 'PCaaC364', 'PCaaC366', 'PCaaC383','PCaaC384', 'PCaaC402', 'PCaaC406', 'PCaaC421', 'PCaaC422', 'PCaaC425', 'PCaaC426', 'PCaeC300', 'PCaeC302', 'PCaeC321', 'PCaeC322', 'PCaeC340', 'PCaeC342', 'PCaeC343', 'PCaeC362', 'PCaeC363', 'PCaeC380', 'PCaeC386', 'PCaeC420', 'PCaeC445', 'PCaeC446', 'Phe', 'Pro', 'SBP', 'SDMA', 'SMC161', 'SMC181', 'SMC202', 'SMC241', 'SMC260', 'SMC261', 'SMOHC141', 'SMOHC161', 'SMOHC241', 'Sarcosine', 'Ser', 'TG', 'Trp', 'Tyr', 'UA', 'Val', 'age', 'lysoPCaC161', 'lysoPCaC182', 'lysoPCaC203', 'lysoPCaC204', 'lysoPCaC260'}
len(set_RF_save)
```




    95




```python
set_XGB_save={'AcSugar', 'Ala', 'Alcohol', 'Arg', 'Asn', 'Asp', 'BMI', 'BW', 'C0', 'C10', 'C101', 'C102', 'C12', 'C121', 'C141', 'C16', 'C161', 'C16OH', 'C18', 'C181', 'C182', 'C2', 'C3', 'C4', 'C51', 'C51DC', 'C6C41DC', 'C7DC', 'C8', 'CHOL', 'Cit', 'Creatinine', 'Creatinine_MS', 'DBP', 'DM', 'Gln', 'Glu', 'Gly', 'HDL', 'HT', 'HTN_drug', 'HeartRate', 'Hip', 'His', 'Homocysteine', 'HsCRP', 'Ile', 'Kynurenine', 'LDL', 'Lipid_drug', 'MBP', 'Met', 'Orn', 'PCaaC281', 'PCaaC300', 'PCaaC323', 'PCaaC341', 'PCaaC342', 'PCaaC360', 'PCaaC364', 'PCaaC365', 'PCaaC366', 'PCaaC383', 'PCaaC384', 'PCaaC385', 'PCaaC403', 'PCaaC404', 'PCaaC406', 'PCaaC422', 'PCaeC300', 'PCaeC322', 'PCaeC340', 'PCaeC342', 'PCaeC343', 'PCaeC360', 'PCaeC361', 'PCaeC363', 'PCaeC380', 'PCaeC382', 'PCaeC385', 'PCaeC386', 'PCaeC420', 'PCaeC421', 'PCaeC423', 'PCaeC446', 'Phe', 'Pro', 'SBP', 'SDMA', 'SMC180', 'SMC181', 'SMC202', 'SMC240', 'SMC241', 'SMC260', 'SMC261', 'SMOHC141', 'SMOHC161', 'SMOHC241', 'Sarcosine', 'Ser', 'Smoking', 'TG', 'Trp', 'UA', 'Val', 'age', 'lysoPCaC161', 'lysoPCaC203', 'lysoPCaC204', 'lysoPCaC240', 'lysoPCaC260', 'lysoPCaC261', 'lysoPCaC281','t4OHPro'}
len(set_XGB_save)
```




    115




```python
set_GBC_save={'ADMA', 'AcSugar', 'Ala', 'Arg', 'Asn', 'Asp', 'BH', 'BMI', 'BW', 'C0', 'C10', 'C101', 'C102', 'C12', 'C121', 'C12DC', 'C141', 'C161', 'C161OH', 'C162OH', 'C18', 'C182', 'C2', 'C3', 'C3DCC4OH', 'C3OH', 'C4', 'C5', 'C51', 'C51DC', 'C5DCC6OH', 'C61', 'C7DC', 'C8', 'CHOL', 'Cit', 'Creatinine', 'Creatinine_MS', 'DBP', 'DM', 'DM_drug', 'Gln', 'Glu', 'Gly', 'HDL', 'HT', 'HTN_drug', 'Hip', 'HsCRP', 'Ile', 'Kynurenine', 'LDL', 'Leu', 'Lipid_drug', 'Lys', 'Met', 'Orn', 'PCaaC240', 'PCaaC300', 'PCaaC323', 'PCaaC341', 'PCaaC342', 'PCaaC361', 'PCaaC363', 'PCaaC364', 'PCaaC366', 'PCaaC380', 'PCaaC383', 'PCaaC384', 'PCaaC385', 'PCaaC402', 'PCaaC404', 'PCaaC406', 'PCaeC300', 'PCaeC321', 'PCaeC340', 'PCaeC342', 'PCaeC343', 'PCaeC360', 'PCaeC361', 'PCaeC362', 'PCaeC380', 'PCaeC382', 'PCaeC383', 'PCaeC385', 'PCaeC386', 'PCaeC402', 'PCaeC420', 'PCaeC421', 'PCaeC422', 'PCaeC423', 'PCaeC425', 'PCaeC446', 'Phe', 'Pro', 'SBP', 'SDMA', 'SMC180', 'SMC181', 'SMC202', 'SMC240', 'SMC241', 'SMC260', 'SMC261', 'SMOHC161', 'SMOHC222', 'SMOHC241', 'Sarcosine', 'Ser', 'Smoking', 'TG', 'Trp', 'UA', 'Val', 'age', 'alphaAAA', 'lysoPCaC161', 'lysoPCaC182', 'lysoPCaC203', 'lysoPCaC204', 'lysoPCaC240', 'lysoPCaC260','t4OHPro'}
len(set_GBC_save)
```




    123




```python
feature_6_models_save={'Trp', 'Kynurenine', 'BMI', 'C182', 'C4', 'HTN_drug'}
len(feature_6_models_save)
```




    6




```python
feature_5_models=list(set(set_LR_save) & set(set_SVC_save) & set(set_RF_save) & set(set_XGB_save) & set(set_GBC_save))
len(feature_5_models)
```




    27




```python
#feature_5_models
```


```python
from venn import venn
from venn import pseudovenn
import matplotlib

%matplotlib inline

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
```


    
![png](output_120_0.png)
    



```python
df_LR_save=pd.DataFrame(set_LR_save)
df_SVC_save=pd.DataFrame(set_SVC_save)
df_DT_save=pd.DataFrame(set_DT_save)
df_RF_save=pd.DataFrame(set_RF_save)
df_XGB_save=pd.DataFrame(set_XGB_save)
df_GBC_save=pd.DataFrame(set_GBC_save)
df_feature_5_models=pd.DataFrame(feature_5_models)
```


```python
df_all_save=pd.concat([df_LR_save.reset_index(drop=True), df_SVC_save,df_DT_save,df_RF_save,df_XGB_save,df_GBC_save,df_feature_5_models], axis=1)
```


```python
df_all_save
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>0</th>
      <th>0</th>
      <th>0</th>
      <th>0</th>
      <th>0</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BW</td>
      <td>C7DC</td>
      <td>Trp</td>
      <td>PCaaC366</td>
      <td>PCaaC366</td>
      <td>PCaaC366</td>
      <td>BW</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DM</td>
      <td>lysoPCaC204</td>
      <td>PCaeC361</td>
      <td>UA</td>
      <td>UA</td>
      <td>UA</td>
      <td>Pro</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pro</td>
      <td>Pro</td>
      <td>HsCRP</td>
      <td>C7DC</td>
      <td>C7DC</td>
      <td>C7DC</td>
      <td>HDL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HDL</td>
      <td>Asn</td>
      <td>Val</td>
      <td>lysoPCaC204</td>
      <td>HeartRate</td>
      <td>lysoPCaC204</td>
      <td>PCaaC364</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Asn</td>
      <td>PCaaC364</td>
      <td>C182</td>
      <td>Pro</td>
      <td>lysoPCaC204</td>
      <td>PCaeC422</td>
      <td>HTN_drug</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>118</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Asp</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>119</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Phe</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>120</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>lysoPCaC203</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>121</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>PCaaC384</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>122</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CHOL</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>123 rows × 7 columns</p>
</div>




```python
df_all_save.columns = ['Logistic_Regression', 'SVM','DecisionTree','RandomForest','XGBoost','GradientBoost','5_models_overlap']
```


```python
df_all_save.to_csv("./new-data-220328/RFECV_gridscores//RFECV_features_all.txt",sep="\t")
```

# Select features by different methods (by RFE method)


```python
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
```

    Overlap features of 6 models is: {'Trp', 'C4', 'HTN_drug', 'BMI', 'Kynurenine', 'C182'}
    Optimal N of LR features is: 62
    Optimal N of SVC features is: 83
    Optimal N of DT features is: 183
    Optimal N of RF features is: 183
    Optimal N of XGB features is: 115
    Optimal N of GBC features is: 156
    Optimal AUC of LR features is : 0.901003
    Optimal AUC of SVC features is : 0.883022
    Optimal AUC of DT features is : 0.726200
    Optimal AUC of RF features is : 0.891653
    Optimal AUC of XGB features is : 0.871323
    Optimal AUC of GBC features is : 0.889249



```python
# set_LR
# set_SVC
# set_DT
# set_RF
# set_XGB
# set_GBC
# feature_6_models
```

# Get Ready data (After model feature selection)


```python
set_LR={'Ala','Alcohol','Asn','BMI','BW','C0', 'C10', 'C12DC', 'C141','C142OH','C182','C2','C3','C4','C41','C5','C6C41DC',
        'C8','CHOL','Cit','Creatinine','DBP','DM','DM_drug','HDL','HT','HTN_drug','Kynurenine','LDL','Lipid_drug','Lys',
        'PCaaC300','PCaaC360','PCaaC364','PCaaC365','PCaaC404','PCaaC406','PCaaC422','PCaaC424','PCaeC302','PCaeC342','PCaeC360',
        'PCaeC362','PCaeC383','PCaeC405','PCaeC424','Phe','Pro','SMC180','SMC181','SMC241','SMOHC161','SMOHC221','SMOHC222',
        'Sarcosine','Ser','Smoking','Thr','Trp','age','lysoPCaC260', 'lysoPCaC281'}
```


```python
feature_6_models={'Trp', 'Kynurenine', 'BMI', 'C182', 'C4', 'HTN_drug'}
```


```python
feature_5_models={'BW', 'Pro', 'HDL', 'PCaaC364', 'HTN_drug', 'C4', 'Ser', 'Lipid_drug', 'LDL', 'C141', 'Creatinine', 'Trp', 'C0', 'PCaaC406', 'SMOHC161', 'Sarcosine', 'C3', 'C10', 'DBP', 'C182', 'C8', 'age', 'Phe', 'BMI', 'Kynurenine', 'DM', 'CHOL'}
```


```python
len(set_LR),len(feature_6_models),len(feature_5_models)
```




    (62, 6, 27)




```python
import pandas as pd
pd.options.mode.chained_assignment = None
```


```python
X_train_final=X_train[feature_5_models]
X_test_final=X_test[feature_5_models]
y_train_final=y_train
y_test_final=y_test
```


```python
#X_train_final
```


```python
X_train_final.columns.array
```




    <PandasArray>
    [  'PCaaC406',        'Trp',   'HTN_drug',         'DM',         'C3',
             'C8', 'Creatinine',        'BMI',  'Sarcosine',        'Phe',
     'Lipid_drug',        'Pro',        'age',         'C4',       'C141',
            'HDL',       'C182',       'CHOL',        'LDL',   'PCaaC364',
             'C0',   'SMOHC161',        'C10', 'Kynurenine',        'DBP',
            'Ser',         'BW']
    Length: 27, dtype: object



# Receiver Operating Characteristic (ROC) with cross validation  (After model feature selection)


```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
```


```python
from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
```


```python
X_cv_final = X_train_final
y_cv_final = y_train_final
```


```python
X_cv_nmp_final=X_cv_final.to_numpy()
y_cv_nmp_final=y_cv_final.to_numpy()
```


```python
len(X_cv_nmp_final),len(y_cv_nmp_final)
```




    (287, 287)



# Logistic Regression (After model feature selection)


```python
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
```


    
![png](output_145_0.png)
    



```python
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
```

    10 fold Accuracy: 0.82 (± 0.16)
    10 fold AUC: 0.93 (± 0.10)
    10 fold Recall: 0.80 (± 0.22)
    10 fold Precision: 0.81 (± 0.17)
    10 fold f1: 0.81 (± 0.17)



```python
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
```

    Accuracy (overall correct predictions): 0.78
    Auc: 0.9
    Recall (all 1s predicted right): 0.72
    Precision (confidence when predicting a 1): 0.85
    Detail:
                  precision    recall  f1-score   support
    
               0       0.71      0.84      0.77        32
               1       0.85      0.72      0.78        40
    
        accuracy                           0.78        72
       macro avg       0.78      0.78      0.78        72
    weighted avg       0.79      0.78      0.78        72
    



    
![png](output_147_1.png)
    



```python
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
```


    
![png](output_148_0.png)
    


    Best Threshold=-2.185215, F-Score=0.854
    =====================================
    When Threshold = 0: (defalt)
    When Threshold=0, the confusion matrix is:
     [[27  5]
     [11 29]]
    When Threshold=0, the precision is: 0.8529411764705882
    When Threshold=0, the recall is: 0.725
    =====================================
    When Threshold = -2.185215:
    When Threshold= -1.635653, the confusion matrix is:
     [[24  8]
     [ 6 34]]
    When Threshold= -1.635653, the precision is: 0.8095238095238095
    When Threshold= -1.635653, the recall is: 0.85


# SVM Classifier (After model feature selection)


```python
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
```


    
![png](output_150_0.png)
    



```python
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
```

    10 fold Accuracy: 0.83 (± 0.09)
    10 fold AUC: 0.91 (± 0.07)
    10 fold Recall: 0.85 (± 0.16)
    10 fold Precision: 0.81 (± 0.10)
    10 fold f1: 0.83 (± 0.10)



```python
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
```

    Accuracy (overall correct predictions): 0.83
    Auc: 0.9
    Recall (all 1s predicted right): 0.75
    Precision (confidence when predicting a 1): 0.94
    Detail:
                  precision    recall  f1-score   support
    
               0       0.75      0.94      0.83        32
               1       0.94      0.75      0.83        40
    
        accuracy                           0.83        72
       macro avg       0.84      0.84      0.83        72
    weighted avg       0.85      0.83      0.83        72
    



    
![png](output_152_1.png)
    


# Decision Tree Classifier (After model feature selection)


```python
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
```


    
![png](output_154_0.png)
    



```python
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
```

    10 fold Accuracy: 0.69 (± 0.14)
    10 fold AUC: 0.66 (± 0.17)
    10 fold Recall: 0.63 (± 0.19)
    10 fold Precision: 0.70 (± 0.23)
    10 fold f1: 0.66 (± 0.15)



```python
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
```

    Accuracy (overall correct predictions): 0.68
    Auc: 0.62
    Recall (all 1s predicted right): 0.6
    Precision (confidence when predicting a 1): 0.77
    Detail:
                  precision    recall  f1-score   support
    
               0       0.61      0.78      0.68        32
               1       0.77      0.60      0.68        40
    
        accuracy                           0.68        72
       macro avg       0.69      0.69      0.68        72
    weighted avg       0.70      0.68      0.68        72
    



    
![png](output_156_1.png)
    



```python
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
```

    |--- HTN_drug <= 0.50
    |   |--- HDL <= -0.22
    |   |   |--- BMI <= 0.66
    |   |   |   |--- LDL <= 1.09
    |   |   |   |   |--- C141 <= 0.91
    |   |   |   |   |   |--- Trp <= 0.02
    |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- Trp >  0.02
    |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |--- C141 >  0.91
    |   |   |   |   |   |--- class: 0
    |   |   |   |--- LDL >  1.09
    |   |   |   |   |--- class: 0
    |   |   |--- BMI >  0.66
    |   |   |   |--- Creatinine <= 0.24
    |   |   |   |   |--- class: 0
    |   |   |   |--- Creatinine >  0.24
    |   |   |   |   |--- Ser <= -0.17
    |   |   |   |   |   |--- class: 1
    |   |   |   |   |--- Ser >  -0.17
    |   |   |   |   |   |--- class: 0
    |   |--- HDL >  -0.22
    |   |   |--- Kynurenine <= 1.15
    |   |   |   |--- C182 <= -1.01
    |   |   |   |   |--- C10 <= -0.69
    |   |   |   |   |   |--- class: 1
    |   |   |   |   |--- C10 >  -0.69
    |   |   |   |   |   |--- class: 0
    |   |   |   |--- C182 >  -1.01
    |   |   |   |   |--- BMI <= -1.55
    |   |   |   |   |   |--- PCaaC406 <= 0.13
    |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- PCaaC406 >  0.13
    |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |--- BMI >  -1.55
    |   |   |   |   |   |--- DBP <= -1.39
    |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- DBP >  -1.39
    |   |   |   |   |   |   |--- class: 0
    |   |   |--- Kynurenine >  1.15
    |   |   |   |--- C141 <= 0.38
    |   |   |   |   |--- class: 1
    |   |   |   |--- C141 >  0.38
    |   |   |   |   |--- BMI <= -0.50
    |   |   |   |   |   |--- class: 1
    |   |   |   |   |--- BMI >  -0.50
    |   |   |   |   |   |--- class: 0
    |--- HTN_drug >  0.50
    |   |--- C4 <= -0.62
    |   |   |--- C3 <= -1.10
    |   |   |   |--- class: 1
    |   |   |--- C3 >  -1.10
    |   |   |   |--- C182 <= -0.24
    |   |   |   |   |--- class: 0
    |   |   |   |--- C182 >  -0.24
    |   |   |   |   |--- class: 1
    |   |--- C4 >  -0.62
    |   |   |--- Kynurenine <= -1.05
    |   |   |   |--- C4 <= 0.38
    |   |   |   |   |--- class: 0
    |   |   |   |--- C4 >  0.38
    |   |   |   |   |--- class: 1
    |   |   |--- Kynurenine >  -1.05
    |   |   |   |--- BW <= 1.51
    |   |   |   |   |--- HDL <= 2.47
    |   |   |   |   |   |--- BMI <= 1.60
    |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- BMI >  1.60
    |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |--- HDL >  2.47
    |   |   |   |   |   |--- class: 0
    |   |   |   |--- BW >  1.51
    |   |   |   |   |--- C3 <= 0.56
    |   |   |   |   |   |--- class: 0
    |   |   |   |   |--- C3 >  0.56
    |   |   |   |   |   |--- class: 1
    



    
![png](output_157_1.png)
    


# Random Forest Classifier (After model feature selection)


```python
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
```


    
![png](output_159_0.png)
    



```python
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
```

    10 fold Accuracy: 0.82 (± 0.22)
    10 fold AUC: 0.90 (± 0.14)
    10 fold Recall: 0.78 (± 0.29)
    10 fold Precision: 0.82 (± 0.26)
    10 fold f1: 0.80 (± 0.25)



```python
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
```

    Accuracy (overall correct predictions): 0.81
    Auc: 0.88
    Recall (all 1s predicted right): 0.78
    Precision (confidence when predicting a 1): 0.86
    Detail:
                  precision    recall  f1-score   support
    
               0       0.75      0.84      0.79        32
               1       0.86      0.78      0.82        40
    
        accuracy                           0.81        72
       macro avg       0.81      0.81      0.80        72
    weighted avg       0.81      0.81      0.81        72
    



    
![png](output_161_1.png)
    



```python
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

```

    Elapsed time to compute the importances: 0.011 seconds





    <AxesSubplot:title={'center':'Feature Importance'}>




    
![png](output_162_2.png)
    


# XGB Classifier (After model feature selection)


```python
#list(X_train_final.columns.values)
```


```python
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
```


    
![png](output_165_0.png)
    



```python
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
```

    10 fold Accuracy: 0.81 (± 0.16)
    10 fold AUC: 0.89 (± 0.10)
    10 fold Recall: 0.76 (± 0.28)
    10 fold Precision: 0.83 (± 0.19)
    10 fold f1: 0.79 (± 0.19)



```python
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
```

    Accuracy (overall correct predictions): 0.82
    Auc: 0.9
    Recall (all 1s predicted right): 0.75
    Precision (confidence when predicting a 1): 0.91
    Detail:
                  precision    recall  f1-score   support
    
               0       0.74      0.91      0.82        32
               1       0.91      0.75      0.82        40
    
        accuracy                           0.82        72
       macro avg       0.83      0.83      0.82        72
    weighted avg       0.84      0.82      0.82        72
    



    
![png](output_167_1.png)
    


# Gradient Boost (After model feature selection)


```python
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
```


    
![png](output_169_0.png)
    



```python
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
```

    10 fold Accuracy: 0.81 (± 0.13)
    10 fold AUC: 0.90 (± 0.11)
    10 fold Recall: 0.79 (± 0.25)
    10 fold Precision: 0.81 (± 0.16)
    10 fold f1: 0.80 (± 0.16)



```python
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
```

    Accuracy (overall correct predictions): 0.75
    Auc: 0.86
    Recall (all 1s predicted right): 0.65
    Precision (confidence when predicting a 1): 0.87
    Detail:
                  precision    recall  f1-score   support
    
               0       0.67      0.88      0.76        32
               1       0.87      0.65      0.74        40
    
        accuracy                           0.75        72
       macro avg       0.77      0.76      0.75        72
    weighted avg       0.78      0.75      0.75        72
    



    
![png](output_171_1.png)
    



```python
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
```


    
![png](output_172_0.png)
    


# Final All features form RFE


```python
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
```

    Overlap features of 5 models is: ['BW', 'Pro', 'HDL', 'PCaaC364', 'HTN_drug', 'C4', 'Ser', 'Lipid_drug', 'LDL', 'C141', 'Creatinine', 'Trp', 'C0', 'PCaaC406', 'SMOHC161', 'Sarcosine', 'C3', 'C10', 'DBP', 'C182', 'C8', 'age', 'Phe', 'BMI', 'Kynurenine', 'DM', 'CHOL'] 
    Optimal N of LR features is: 62
    Optimal N of SVC features is: 83
    Optimal N of DT features is: 183
    Optimal N of RF features is: 183
    Optimal N of XGB features is: 115
    Optimal N of GBC features is: 156
    Optimal AUC of LR features is : 0.901003
    Optimal AUC of SVC features is : 0.883022
    Optimal AUC of DT features is : 0.726200
    Optimal AUC of RF features is : 0.891653
    Optimal AUC of XGB features is : 0.871323
    Optimal AUC of GBC features is : 0.889249



```python

```
