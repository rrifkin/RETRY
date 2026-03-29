#! /usr/bin/env python

import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

def retry_stats (data_file, dict_file):
	
	# Parse data and dictionary files and return patient and eeg dataframes
	patient_data, eeg_data = redcap_parser(data_file, dict_file)

	# Parse dataframes to find high-level patient characteristics  
	patient_data = find_hyperexcitability (patient_data, eeg_data)
	patient_data = find_ever_sz (patient_data)

	# Descriptive statistics
	avg_patient_age = patient_data['age'].mean()
	print("\nAverage patient age =", avg_patient_age, "\n")

	percent_female = (len(patient_data[patient_data['sex'] == "Female"]) / len(patient_data) *100)
	print("Percent Female:", percent_female, "\n")
	
	age_by_grade = patient_data.groupby('who')['age'].mean()
	print("Patient age by WHO tumor grade:")
	print(age_by_grade, "\n")

	preop_sz_by_grade = patient_data.groupby('who')['preop_sz'].value_counts().unstack()
	print("Pre-operative seizures by WHO tumor grade:")
	print(preop_sz_by_grade, "\n")
	
	late_sz_by_grade = patient_data.groupby('who')['late_sz'].value_counts().unstack()
	print("Late-onset seizures by WHO tumor grade:")
	print(late_sz_by_grade, "\n")
	
	preop_sz_by_idh = patient_data.groupby('idh')['preop_sz'].value_counts().unstack()
	print("Pre-operative seizures by IDH mutation status:")
	print(preop_sz_by_idh, "\n")

	patient_data.to_csv("~/Desktop/patient_data.csv",encoding="utf-8-sig")
	eeg_data.to_csv("~/Desktop/eeg_data.csv", encoding="utf-8-sig")

	# Logistic regression to identify predictors of late-onset seizures in patients without seizures at baseline
	print("In patients without seizures at baseline, what factors predict late_onset seizures?\n")
	patients_wo_baseline_sz = patient_data[patient_data['preop_sz'] != "No"]
	x = patients_wo_baseline_sz[['hyperexcitability', 'idh', 'who', 'age']].copy()
	# x['idh'] = x['idh'].map({'Yes': 1, 'No': 0})
	# x['who'] = x['who'].map({'1': 1, '2': 2, '3': 3, '4': 4, 'Indeterminate': np.nan})
	y = patients_wo_baseline_sz.loc[:, 'late_sz'].map({'Yes': 1, 'No': 0})

	# Drop rows where either x or y has NaN
	valid_idx = x.dropna().index.intersection(y.dropna().index)
	x = x.loc[valid_idx]
	y = y.loc[valid_idx]	
	x = sm.add_constant(x)

	model = sm.Logit(y,x).fit()
	print(model.summary())

	params = model.params
	conf = model.conf_int()
	odds_ratios = np.exp(params)
	odds_ci = np.exp(conf)

	print(pd.DataFrame({
		"OR": odds_ratios,
		"2.5%": odds_ci[0],
		"97.5%": odds_ci[1]
	}))

	# Ordinal logistic regression to identify predictors of WHO tumor grade
	print("What factors predict WHO tumor grade?\n")
	x = patient_data[['lpd', 'gpd', 'bipd', 'lrda', 'esz', 'ied', 'idh', 'age']].copy()
	# x['idh'] = x['idh'].map({'Yes': 1, 'No': 0})
	y = patient_data.loc[:, 'who'] #.map({'1': 1, '2': 2, '3': 3, '4': 4, 'Indeterminate': np.nan}).astype(float)

	# Drop rows where either x or y has NaN
	valid_idx = x.dropna().index.intersection(y.dropna().index)
	x = x.loc[valid_idx]
	y = y.loc[valid_idx]	
	x = sm.add_constant(x)

	model = sm.MNLogit(y, x).fit()
	print(model.summary())

	params = model.params
	odds_ratios = np.exp(params)

	print(pd.DataFrame(odds_ratios, columns=['OR']))


def redcap_parser (data_file, dict_file):

	# Import CSV files as DataFrames
	raw_data = pd.read_csv(data_file, dtype=str)				# Raw patient and EEG data exported as CSV from RedCap
	data_dict = pd.read_csv(dict_file, dtype=str)				# Data Dictionary exported as CSV file from RedCap

	# Create separate dataframes of radio/dropdown and checkbox Data Dictionary entries
	multchoice_dict = data_dict[(data_dict['Field Type'] == 'radio')|(data_dict['Field Type'] == 'dropdown')]
	checkbox_dict = data_dict[(data_dict['Field Type'] == 'checkbox')]

	# Convert df column to an actual Python dictionary and append it as another column
	multchoice_dict.loc[:,'parsed_dict'] = multchoice_dict.loc[:,'Choices, Calculations, OR Slider Labels'].apply(convert_to_dict)
	checkbox_dict.loc[:,'parsed_dict'] = checkbox_dict.loc[:,'Choices, Calculations, OR Slider Labels'].apply(convert_to_dict)
	
	# In df containing patient/EEG data, replace meaningless numbers with their meaning using dictionary
	multchoice_cleaned_data = replace_with_dict(raw_data, multchoice_dict)
	cleaned_data = replace_with_dict_checkboxes(multchoice_cleaned_data, checkbox_dict)

	# create a dataframe of just the patient data
	patient_data = cleaned_data[cleaned_data['redcap_repeat_instrument'] != 'eeg_data']
	patient_data = patient_data.loc[:,'study_id':'glioma_complete']

	# create a dataframe of just the EEG data
	eeg_data = cleaned_data[cleaned_data['redcap_repeat_instrument'] == 'eeg_data']
	eeg_data_1 = eeg_data.loc[:,'study_id':'redcap_repeat_instance']
	eeg_data_2 = eeg_data.loc[:,'eeg_inclusion':'eeg_data_complete']
	eeg_data = pd.concat([eeg_data_1,eeg_data_2], axis=1)

	# clean up data types in patient data
	patient_data["age"] = patient_data["age"].astype(float)
	patient_data["study_id"] = patient_data["study_id"].astype(int)
	patient_data['idh'] = patient_data['idh'].map({'Yes': 1, 'No': 0})
	patient_data['who'] = patient_data['who'].map({'1': 1, '2': 2, '3': 3, '4': 4, 'Indeterminate': np.nan})
	eeg_data["study_id"] = eeg_data["study_id"].astype(int)

	return (patient_data, eeg_data)

def convert_to_dict (string):
	return {pair.split(', ')[0]: pair.split(', ')[1].strip() for pair in string.split(' | ')}

# replaces radio / dropdown meaningless numbers with their meaning using a dictionary
def replace_with_dict(raw_data, multchoice_dict):
	raw_data = raw_data.copy()
	for index, row in multchoice_dict.iterrows():
		column_name = row['Variable / Field Name']
		replacement_dict = row['parsed_dict']
		
		if column_name in raw_data.columns:
			raw_data[column_name] = raw_data[column_name].replace(replacement_dict)

	#renaming this variable because it's really confusing
	raw_data.rename(columns={'any_sz': 'late_sz'}, inplace=True)

	return raw_data

# replaces checkbox meaningless numbers with their meaning using a dictionary
def replace_with_dict_checkboxes(raw_data, checkbox_dict):
	raw_data = raw_data.copy()
	for index, row in checkbox_dict.iterrows():
		column_name = row['Variable / Field Name']
		replacement_dict = row['parsed_dict']

		if column_name in raw_data.columns:
			output = []
			for i, this_item in raw_data[column_name].items():
				this_string =str(this_item)
				result = ",".join(replacement_dict.get(item.strip(), item.strip()) for item in this_string.split(","))
				output.append(result)
			raw_data[column_name] = output

	return (raw_data)

def find_hyperexcitability (patient_data, eeg_data):
	patient_data = patient_data.copy()
	eeg_data = eeg_data.copy()

	gpd = eeg_data.loc[eeg_data['gpd_yn'] == "1", 'study_id'].drop_duplicates()
	patient_data['gpd'] = patient_data['study_id'].isin(gpd).astype(int)

	lpd = eeg_data.loc[eeg_data['lpd_yn'] == "1", 'study_id'].drop_duplicates()
	patient_data['lpd'] = patient_data['study_id'].isin(lpd).astype(int)

	bipd = eeg_data.loc[eeg_data['bipd_yn'] == "1", 'study_id'].drop_duplicates()
	patient_data['bipd'] = patient_data['study_id'].isin(bipd).astype(int)

	lrda = eeg_data.loc[eeg_data['lrda_yn'] == "1", 'study_id'].drop_duplicates()
	patient_data['lrda'] = patient_data['study_id'].isin(lrda).astype(int)

	esz = eeg_data.loc[eeg_data['sz_yn'] == "1", 'study_id'].drop_duplicates()
	patient_data['esz'] = patient_data['study_id'].isin(esz).astype(int)

	ied = eeg_data.loc[eeg_data['epi_disch_yn'] == "1", 'study_id'].drop_duplicates()
	patient_data['ied'] = patient_data['study_id'].isin(ied).astype(int)

	hyperexcitable = pd.concat([gpd, lpd, bipd, lrda, esz, ied], ignore_index=True)
	patient_data['hyperexcitability'] = patient_data['study_id'].isin(hyperexcitable).astype(int)

	return (patient_data)

# determines if patient ever had a clinical seizure
def find_ever_sz (patient_data):
	patient_data = patient_data.copy()

	ever_sz = patient_data.loc[(patient_data['preop_sz'] == "Yes") | (patient_data['late_sz'] == "Yes")]

	patient_data['ever_sz'] = patient_data['study_id'].isin(ever_sz['study_id']).astype(int)

	return (patient_data)

if __name__ == "__main__":
	import sys
	retry_stats (str(sys.argv[1]), str(sys.argv[2]))
