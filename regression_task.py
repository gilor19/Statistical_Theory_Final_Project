import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from matplotlib import pyplot


# arranging the csv
df = pd.read_csv('kaggle_survey_2022_responses.csv', low_memory=False)
# chosing the columns we need:
df = df.loc[:, ['Q2','Q3','Q4','Q5','Q8','Q9','Q11','Q16', 'Q23', 'Q24', 'Q29']]
# deleting the questios row:
df.drop(axis=0, index=0, inplace=True)
# setting the column names:
df.columns = ['age', 'gender', 'country', 'student', 'level_on2Y', 'published_reaerch', 'experience_programming',
              'experience_ML', 'title', 'industry', 'salary']

# Drop rows with missing target values (salary)
df.dropna(subset=['salary'], inplace=True)

# remove rows that do not have the gender specified as either man or women.
df = df.loc[df['gender'].isin(['Man', 'Woman'])]

df = df.loc[df['salary'].isin(['5,000-7,499', '7,500-9,999', '10,000-14,999', '15,000-19,999',
                               '20,000-24,999', '25,000-29,999', '30,000-39,999'])]


# split data to features and target
Y = df[['salary']]
X = df.drop(['salary'], axis=1)

# split data to train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7) # same distribution for women in both train and test

# # validate train and test have the same gender distribution
# X_train['gender'].value_counts().plot(kind='bar')
# X_test['gender'].value_counts().plot(kind='bar')

# fill nulls
for col in X_train.columns:
  X_train[col] = X_train[col].fillna(X_train[col].mode()[0])
  X_test[col] = X_test[col].fillna(X_train[col].mode()[0])

# convert categorical data to one-hot
boolean_columns = ['student', 'published_reaerch', 'gender']
categorical_columns = ['age', 'country', 'level_on2Y', 'experience_programming', 'experience_ML', 'title', 'industry']

X_train = pd.get_dummies(X_train, columns=boolean_columns, drop_first=True)
X_train = pd.get_dummies(X_train, columns=categorical_columns, drop_first=False)

X_test = pd.get_dummies(X_test, columns=boolean_columns, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_columns, drop_first=False)


# compare columns between train and test
for col in X_train.columns:
    if col not in X_test.columns:
        X_test[col] = 0

    for col in X_test.columns:
      if col not in X_train.columns:
        X_train[col] = 0
X_test = X_test[X_train.columns]

#convert target column to numeric
labels = {'$0-999': np.log(500), '1,000-1,999': np.log(1500), '2,000-2,999': np.log(2500), '3,000-3,999': np.log(3500),
          '4,000-4,999': np.log(4500), '5,000-7,499': np.log(6250), '7,500-9,999': np.log(8750),
          '10,000-14,999': np.log(12500), '15,000-19,999': np.log(17500), '20,000-24,999': np.log(22500),
          '25,000-29,999': np.log(27500), '30,000-39,999': np.log(35000), '40,000-49,999': np.log(45000),
          '50,000-59,999': np.log(55000), '60,000-69,999': np.log(65000), '70,000-79,999': np.log(75000),
          '80,000-89,999': np.log(85000), '90,000-99,999': np.log(95000), '100,000-124,999': np.log(112500),
          '125,000-149,999': np.log(137500), '150,000-199,999': np.log(175000), '200,000-249,999': np.log(225000),
          '250,000-299,999': np.log(275000),  '300,000-499,999': np.log(400000), '$500,000-999,999': np.log(750000),
          '>$1,000,000': np.log(1000000)}

Y_train['salary'] = Y_train['salary'].map(labels)
Y_test['salary'] = Y_test['salary'].map(labels)

# some final adjustments are required for XGBoost.
X_train['experience_programming_less_1 years'] = X_train['experience_programming_< 1 years']
X_train = X_train.drop(['experience_programming_< 1 years'], axis=1)

X_test['experience_programming_less_1 years'] = X_test['experience_programming_< 1 years']
X_test = X_test.drop(['experience_programming_< 1 years'], axis=1)

# train model
evalset = [(X_train, Y_train)]
model = xgb.XGBRegressor()
model.fit(X_train, Y_train, eval_set=evalset)
results = model.evals_result()
pyplot.plot(results['validation_0']['rmse'], label='train')

# predict and print accuracy
y_pred = model.predict(X_test)
mae = mean_squared_error(Y_test, y_pred)
print("mae: " + str(mae))

bad_count = 0
for row in X_test.iterrows():
    orig_gender = row[1]['gender_Woman']
    pred = model.predict(pd.DataFrame(row[1]).transpose())

    modified_row = pd.DataFrame(row[1]).transpose()
    modified_row['gender_Woman'] = modified_row['gender_Woman'].replace(orig_gender, abs(1-orig_gender))
    mod_pred = model.predict(modified_row)

    if orig_gender == 0 and mod_pred < pred:
        bad_count += 1
    if orig_gender == 1 and pred < mod_pred:
        bad_count += 1

print(bad_count/len(X_test))




