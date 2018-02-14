import pandas as pd

GREENHOUSE_CSV_DIR = './greenhouse_csv/'

application_stages = pd.read_csv(GREENHOUSE_CSV_DIR + 'application_stages.csv', encoding="ISO-8859-1")
applications = pd.read_csv(GREENHOUSE_CSV_DIR + 'applications.csv', encoding="ISO-8859-1")
candidates = pd.read_csv(GREENHOUSE_CSV_DIR + 'candidates.csv', encoding="ISO-8859-1")
jobs = pd.read_csv(GREENHOUSE_CSV_DIR + 'jobs.csv', encoding="ISO-8859-1")
scorecard_attributes = pd.read_csv(GREENHOUSE_CSV_DIR + 'scorecard_attributes.csv', encoding="ISO-8859-1")
scorecards = pd.read_csv(GREENHOUSE_CSV_DIR + 'scorecards.csv', encoding="ISO-8859-1")


print('\n\n=== application_stages.csv ===')
print(application_stages[0:10])
print('\n\n=== applications.csv ===')
print(applications[0:10])
print('\n\n=== candidates.csv ===')
print(candidates[0:10])
print('\n\n=== jobs.csv ===')
print(jobs[0:10])
print('\n\n=== scorecard_attributes.csv ===')
print(scorecard_attributes[0:10])
print('\n\n=== scorecards.csv ===')
print(scorecards[0:10])
