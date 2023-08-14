import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#arrange the csv
df = pd.read_csv('kaggle_survey_2022_responses.csv', low_memory=False)
#chosing the columns we need:
df=df.loc[:,['Q2','Q3','Q4','Q5','Q8','Q9','Q11','Q16', 'Q23', 'Q24','Q25', 'Q26', 'Q29']]
#deleting the questios row:
df.drop(axis=0, index=0, inplace=True)
#setting the column names:
df.columns = ['age', 'gender', 'country', 'student','level_on2Y','published_reaerch','experience_programming','experience_ML','title','industry','size_company','size_DS_company','salary']

# Drop rows with missing target values (salary)
df.dropna(subset=['salary'], inplace=True)

# remove rows that do not have the gender specified as either man or women.
df = df.loc[df['gender'].isin(['Man', 'Woman'])]

# df = df.loc[df['salary'].isin(['$0-999', '1,000-1,999','5,000-7,499',
#           '7,500-9,999', '10,000-14,999', '15,000-19,999',
#           '20,000-24,999', '25,000-29,999', '30,000-39,999', '40,000-49,999'])]


# split data to features and target
Y = df[['salary']]
X = df.drop(['salary'], axis=1)

# split data to train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)    # same distrubiton for women in both train and test

# # validate train and test have the same gender distribution
# X_train['gender'].value_counts().plot(kind='bar')
# fill nulls
for col in X_train.columns:
  X_train[col] = X_train[col].fillna(X_train[col].mode()[0])
  X_test[col] = X_test[col].fillna(X_train[col].mode()[0])

# convert categorical data to one-hot
boolean_columns = ['student', 'published_reaerch', 'gender']
catgorical_columns = ['age', 'country', 'level_on2Y', 'experience_programming', 'experience_ML', 'title', 'industry','size_company','size_DS_company']

X_train = pd.get_dummies(X_train, columns=boolean_columns, drop_first=True)
X_train = pd.get_dummies(X_train, columns=catgorical_columns, drop_first=False)

X_test = pd.get_dummies(X_test, columns=boolean_columns, drop_first=True)
X_test = pd.get_dummies(X_test, columns=catgorical_columns, drop_first=False)


# compare columns between train and test
for col in X_train.columns:
    if col not in X_test.columns:
        X_test[col] = 0

    for col in X_test.columns:
      if col not in X_train.columns:
        X_train[col] = 0
X_test = X_test[X_train.columns]

# convert target column to labels

labels = {'$0-999': 0, '1,000-1,999': 0, '2,000-2,999': 0, '3,000-3,999': 0, '4,000-4,999': 0, '5,000-7,499': 0,
          '7,500-9,999': 0, '10,000-14,999': 1, '15,000-19,999': 1, '20,000-24,999': 1, '25,000-29,999': 1,
          '30,000-39,999': 1, '40,000-49,999': 1, '50,000-59,999': 1, '60,000-69,999': 1, '70,000-79,999': 1,
          '80,000-89,999': 1, '90,000-99,999': 2, '100,000-124,999': 2, '125,000-149,999': 2, '150,000-199,999': 2,
          '200,000-249,999': 3,  '250,000-299,999': 3,  '300,000-499,999': 3, '$500,000-999,999': 3, '>$1,000,000': 3}
# labels = {'$0-999': 0, '1,000-1,999': 0,  '5,000-7,499': 0,
#           '7,500-9,999': 0, '10,000-14,999': 0, '15,000-19,999': 0,
#           '20,000-24,999': 1, '25,000-29,999': 1, '30,000-39,999': 1, '40,000-49,999': 1}
Y_train['salary'] = Y_train['salary'].map(labels)
Y_test['salary'] = Y_test['salary'].map(labels)


# some final adjustments are required for XGBoost.
X_train['experience_programming_less_1 years'] = X_train['experience_programming_< 1 years']
X_train = X_train.drop(['experience_programming_< 1 years'], axis=1)

X_test['experience_programming_less_1 years'] = X_test['experience_programming_< 1 years']
X_test = X_test.drop(['experience_programming_< 1 years'], axis=1)

# train model
model = XGBClassifier()
model.fit(X_train, Y_train)

# predict and print accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_pred, Y_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

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




