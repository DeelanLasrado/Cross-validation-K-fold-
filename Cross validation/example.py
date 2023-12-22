import pandas as pd
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/krishnaik06/Types-Of-Cross-Validation/main/cancer_dataset.csv')

print(df.head())

X = df.iloc[:,2:]
y = df.iloc[:,1]

print(X.isnull().sum())

df.drop('Unnamed: 32', axis=1, inplace=True)


'''# Train Test Split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
for i in range(100):
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
                                                      random_state=i)

  model = LogisticRegression()
  model.fit(X_train,y_train)
  result = model.score(X_test, y_test)
  print(result)
# 0.9532163742690059
# 0.9415204678362573
# 0.9239766081871345'''


# K Fold Cross Validation
from sklearn.model_selection import KFold, cross_val_score

k = KFold(5) 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
arr = [5,10,15,20,25,30,40,50,60,70]
for i in arr:
  result = cross_val_score(model, X,y, cv=i)
  # print(result)
  print(f'When k={i}',np.mean(result))
  # When k=10, 0.9367794486215539
  # When k=5, 0.9455364073901569


'''When k=5 0.943766495885732
When k=10 0.9455513784461151
When k=15 0.9455192034139402
When k=20 0.9420566502463055
When k=25 0.938498023715415
When k=30 0.9437621832358672
When k=40 0.9492857142857144
When k=50 0.9474242424242424
When k=60 0.9387037037037036
When k=70 0.9452380952380953'''

# Stratified K-fold Cross validation

from sklearn.model_selection import StratifiedKFold
skfold = StratifiedKFold(n_splits=5)
result = cross_val_score(model,X,y,cv=skfold)
print(np.mean(result))



# LOOCV - Leave One Out CV

from sklearn.model_selection import LeaveOneOut

lc = LeaveOneOut()
result = cross_val_score(model, X,y,cv=lc)
np.mean(result)