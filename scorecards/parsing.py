import pandas as pd

GREENHOUSE_CSV_DIR = './greenhouse_csv/'

## Input
scorecard_attributes = pd.read_csv(GREENHOUSE_CSV_DIR + 'scorecard_attributes.csv', usecols=['Application Id', 'Attribute Name', 'Attribute Rating'], encoding="ISO-8859-1")
scorecards = pd.read_csv(GREENHOUSE_CSV_DIR + 'scorecards.csv', usecols=['Application Id', 'Interview Desc', 'Overall Recommendation'], encoding="ISO-8859-1") # We want the scorecards of the applications who Application_id & Interview Desc = Phone Interview / Recruiting Screen / 2nd Phone Interview

## Output
application_stages = pd.read_csv(GREENHOUSE_CSV_DIR + 'application_stages.csv', usecols=['Application Id', 'Stage Name'], encoding="ISO-8859-1") # whether Stage Name = Offer for each Application ID

## Useless
#applications = pd.read_csv(GREENHOUSE_CSV_DIR + 'applications.csv', encoding="ISO-8859-1")
#candidates = pd.read_csv(GREENHOUSE_CSV_DIR + 'candidates.csv', encoding="ISO-8859-1")
#jobs = pd.read_csv(GREENHOUSE_CSV_DIR + 'jobs.csv', encoding="ISO-8859-1")


df = pd.concat([application_stages, scorecard_attributes, scorecards], axis=0, join='outer')

data = {}

for index, row in df.iterrows():
	application_id = row['Application Id']
	if not data.get(application_id):
		data[application_id] = []
	data[application_id].append(row)
	if index > 1000:
		break

for key, value in data.items():
	if len(value) > 8:
		print(value)
		break
# print(data[list(data.keys())[0]])

# res.loc[res['Stage Name'].isin]


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
