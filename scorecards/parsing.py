import pandas as pd
import math
import numpy as np

GREENHOUSE_CSV_DIR = './greenhouse_csv/'

INTERVIEW_DESCRIPTIONS = {
  'phone_interview': [
    '*2nd Phone Interview - Cat 4 Coding',
    '*2nd Phone Interview - Reverse Shadow',
    '*2nd Phone Screen',
    '*2nd Phone Screen: Cat 2 Coding*',
    '*Phone Interview - Cat 2 Coding',
    '*Phone Interview - Technical Screen',
    '*Phone Screen - Test Search*',
    '*Phone Screen',
    '*Phone Screen*',
    '*Phone Screen: 2nd - Test Search*',
    '*Phone Screen: 2nd*',
    '*Phone Screen: Cat 2 Coding*',
    '*Reverse Shadow Phone Screen*',
    '*Reverse Shadow',
    '2nd Phone Interview - Reverse Shadow',
    '2nd Phone Interview',
    '2nd Phone Screen - Test an Airline',
    '2nd Phone Screen',
    'Phone Interview - Practice',
    'Phone Interview - Reverse Shadow',
    'Phone Interview - Technical Screen',
    'Phone Screen 3',
    'Phone Screen',
    'Reverse Shadow',
  ],
  'recruiting_screen': [
    '*Fit Screen',
    '*Recruiter Screen',
    '*Recruiter: Pre Screen',
    '*Recruiter: Pre Screen*',
    'Fit Screen',
    'Recruiter Screen (Eng Specific)',
    'Recruiter Screen',
    'Recruiter: Pre Screen',
    'Recruiting Screen (Eng Specific)',
    'Resume Review',
  ],
}

ALL_INTERVIEW_DESCRIPTIONS = INTERVIEW_DESCRIPTIONS['phone_interview'] + INTERVIEW_DESCRIPTIONS['recruiting_screen']


# --- Loading files ---
## Input
scorecard_attributes = pd.read_csv(
    GREENHOUSE_CSV_DIR + 'scorecard_attributes.csv',
    usecols=['Application Id', 'Attribute Name', 'Attribute Rating', 'Scorecard Id'],
    encoding="ISO-8859-1"
)

scorecards = pd.read_csv(
    GREENHOUSE_CSV_DIR + 'scorecards.csv',
    usecols=['Application Id', 'Interview Desc', 'Overall Recommendation', 'Scorecard Id'],
    encoding="ISO-8859-1"
) # We want the scorecards of the applications who Application_id & Interview Desc = Phone Interview / Recruiting Screen / 2nd Phone Interview

## Output
application_stages = pd.read_csv(
    GREENHOUSE_CSV_DIR + 'application_stages.csv',
    usecols=['Application Id', 'Stage Name'],
    encoding="ISO-8859-1"
) # whether Stage Name = Offer for each Application ID

## Useless
#applications = pd.read_csv(GREENHOUSE_CSV_DIR + 'applications.csv', encoding="ISO-8859-1")
#candidates = pd.read_csv(GREENHOUSE_CSV_DIR + 'candidates.csv', encoding="ISO-8859-1")
#jobs = pd.read_csv(GREENHOUSE_CSV_DIR + 'jobs.csv', encoding="ISO-8859-1")


# --- Joining data ---
df = pd.concat([application_stages, scorecard_attributes, scorecards], axis=0, join='outer')

# --- Filtering data ---
df = df.loc[
    df['Interview Desc'].isin(ALL_INTERVIEW_DESCRIPTIONS) |
    df['Attribute Name'].notnull()
]


# --- Aggregating data ---
data = {}
for index, row in df.iterrows():
    application_id = row['Application Id']
    if not data.get(application_id):
       data[application_id] = []
    data[application_id].append(row)

# --- Load data in model

interview_desc = set()
overall_recommendation = set()
recommendation_stats_strings = dict()
class Application:
    def __init__(self, series):
        self.series = series
        self.build_attributes()
        self.build_overall_recommendations()

    def build_attributes(self):
        series = [s for s in self.series if isinstance(s['Attribute Name'], str)]
        self.attributes = [(s['Attribute Name'], s['Attribute Rating']) for s in series]

    def build_overall_recommendations(self):
        series = [s
                  for s in self.series
                  if (
                      isinstance(s['Overall Recommendation'], str) and
                      s['Overall Recommendation'] != 'no_decision'
                  )]
        self.overall_recommendations = [(s['Overall Recommendation'], s['Interview Desc']) for s in series]
        recommendation_stats = {'phone_interview': 0, 'recruiting_screen': 0}
        for s in series:
            for type in ('phone_interview', 'recruiting_screen'):
                if s['Interview Desc'] in INTERVIEW_DESCRIPTIONS[type]:
                    recommendation_stats[type] += 1
        recommendation_stats_string = '{}-{}'.format(
            recommendation_stats['phone_interview'],
            recommendation_stats['recruiting_screen'],
        )
        if not recommendation_stats_strings.get(recommendation_stats_string):
            recommendation_stats_strings[recommendation_stats_string] = {
                'count': 0,
                'example': None,
                'pattern': None,
            }
        recommendation_stats_strings[recommendation_stats_string]['count'] += 1
        recommendation_stats_strings[recommendation_stats_string]['example'] = self.overall_recommendations
        recommendation_stats_strings[recommendation_stats_string]['pattern'] = recommendation_stats_string

        for s in series:
            interview_desc.add(s['Interview Desc'])
            overall_recommendation.add(s['Overall Recommendation'])


applications = []
for index, row in data.items():
    applications.append(Application(row))

import json
sorted_recommendations = sorted(recommendation_stats_strings.values(), key=lambda stat: stat['count'], reverse=True)
print(json.dumps(sorted_recommendations, indent=2))

# print(applications[0].attributes)
print(applications[0].overall_recommendations)
print(applications[0].series)



# # --- Display one ---
# # Overall Recommendation: 'yes', 'no', 'strong_yes', 'definitely_not', 'no_decision'
# for key, value in data.items():
#     # for v in value:
#     #     if not np.isnan(v['Attribute Name']):
#     #         # print(repr(v['Attribute Name']), v['Attribute Name'], v['Attribute Name'].__class__, dir(v['Attribute Name'].__class__))
#     #         print(v)
#     #         break
#     if len(value) > 0:
#         print(value)
#         break



# print('\n\n=== application_stages.csv ===')
# print(application_stages[0:10])
# # print('\n\n=== applications.csv ===')
# # print(applications[0:10])
# # print('\n\n=== candidates.csv ===')
# # print(candidates[0:10])
# # print('\n\n=== jobs.csv ===')
# # print(jobs[0:10])
# print('\n\n=== scorecard_attributes.csv ===')
# print(scorecard_attributes[0:10])
# print('\n\n=== scorecards.csv ===')
# print(scorecards[0:10])
