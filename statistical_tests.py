import pandas as pd
from scipy.stats import ttest_ind, kstest, boxcox
import csv

questions_map = {
                    'Q2': 'Age',
                    'Q3': 'Gender',
                    'Q4': 'Country',
                    'Q5': 'Is Student',
                    'Q8': 'Education Level',
                    'Q45': 'Years of Experience',
                    'Q23': 'Job Title',
                    'Q24': 'Industry'
                }


def data_preprocess(data_path):
    df = pd.read_csv(data_path, low_memory=False)
    # aggregate years of experience in programming and years of experience in ML.
    programming_experience_values_map = {'I have never written code': 0, '< 1 years': 1, '1-2 years': 2, '3-5 years': 3,
                                         '5-10 years': 4, '10-20 years': 5, '20+ years': 6}
    ml_experience_values_map = {'I do not use machine learning methods': 0, 'Under 1 year': 1, '1-2 years': 2, '2-3 years': 3,
                                '3-4 years': 3, '4-5 years': 3, '5-10 years': 4, '10-20 years': 5, '20 or more years': 6}
    df['Q11'] = df['Q11'].map(programming_experience_values_map)
    df['Q16'] = df['Q16'].map(ml_experience_values_map)
    df['Q45'] = df[['Q11', 'Q16']].values.max(1)

    reverse_map = {value: key for key, value in programming_experience_values_map.items()}
    df['Q45'] = df['Q45'].map(reverse_map)

    return df


def convert_salary_to_numeric(salary_category):
    if salary_category == '>$1,000,000':
        return 1000000
    else:
        lower_bound, upper_bound = salary_category.replace(",", "").replace("$", "").split("-")
        mid = int(lower_bound) + (int(upper_bound) - int(lower_bound)) / 2
        return mid


def ttest_on_segment(df, segment):
    """ the segment attribute is a dict contains the part of the data we are interested to apply
        the test on. The keys are the attribute names and the values are the attribute's values """

    # From the original data, keep rows of the chosen segment only
    for att, val in segment.items():
        df = df.loc[df[att] == val]

    # Extract the gender and salary columns from the data
    # Q2 - What is your gender?
    # Q29 - What is your current yearly compensation?
    df = df[['Q3', 'Q29']]
    df.columns = ['Gender', 'Salary']

    # delete rows with nulls in at least one of the columns
    df = df.dropna()

    # drop the first row since it is consists the question itself
    df = df.iloc[1:]

    # convert Salary from categories to numbers
    df['Salary'] = df['Salary'].apply(convert_salary_to_numeric)

    # split data to men and women salaries samples
    men_salaries = df.loc[df['Gender'].isin(['Man'])]
    men_salaries = men_salaries[['Salary']].values.reshape(len(men_salaries))
    women_salaries = df.loc[df['Gender'].isin(['Woman'])]
    women_salaries = women_salaries[['Salary']].values.reshape(len(women_salaries))

    # noramlize salaries samples if needed
    ks_statistic, p_value = kstest(men_salaries, 'norm', N=len(men_salaries))
    if p_value < 0.05:
        men_salaries, _ = boxcox(men_salaries)

    ks_statistic, p_value = kstest(women_salaries, 'norm')
    if p_value < 0.05:
        women_salaries, _ = boxcox(women_salaries)

    # apply t-test
    statistic, p_value = ttest_ind(men_salaries, women_salaries, alternative="greater")

    write_results_to_csv(df, segment, p_value, 't-test')


def write_results_to_csv(df, segment, p_value, test):
    segment = {questions_map[key]: val for key,val in segment.items()}
    with open(test + '.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([str(segment),
                         'Men: ' + str(len(df.loc[df['Gender'] == 'Man']))
                         + ', Women: ' + str(len(df.loc[df['Gender'] == 'Woman']))
                            , p_value])


if __name__ == "__main__":
    df_ = data_preprocess("kaggle_survey_2022_responses.csv")
    segments = [{},
                {'Q45': '< 1 years', 'Q4': 'United States of America',  'Q23': 'Data Analyst (Business, Marketing, Financial, Quantitative, etc)'},
                {'Q45': '< 1 years', 'Q4': 'India',  'Q23': 'Data Analyst (Business, Marketing, Financial, Quantitative, etc)'},

                {'Q24': 'Computers/Technology', 'Q4': 'United States of America', 'Q8': 'Doctoral degree'},
                {'Q24': 'Computers/Technology', 'Q4': 'India', 'Q8': 'Doctoral degree'},

                {'Q24': 'Academics/Education', 'Q4': 'United States of America',  'Q23': 'Teacher / professor'},
                {'Q24': 'Academics/Education', 'Q4': 'India',  'Q23': 'Teacher / professor'},

                {'Q45': '5-10 years', 'Q4': 'United States of America', 'Q8': 'Bachelor’s degree'},
                {'Q45': '5-10 years', 'Q4': 'United States of America', 'Q23': 'Data Scientist', 'Q8': 'Master’s degree'}

    ]
    for seg in segments:
        ttest_on_segment(df_, seg)



