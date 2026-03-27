#! /usr/bin/env python

import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

def retry_stats (data_file, dict_file):
	
	multchoice_dict, patient_data, eeg_data = redcap_parser(data_file, dict_file)
	patient_data = find_hyperexcitability (patient_data, eeg_data)
	patient_data = find_ever_sz (patient_data)

	patient_data["age"] = patient_data["age"].astype(float)
	patient_data["study_id"] = patient_data["study_id"].astype(int)
	patient_data['idh'] = patient_data['idh'].map({'Yes': True, 'No': False})
	
	eeg_data["study_id"] = eeg_data["study_id"].astype(int)

	avg_patient_age = patient_data['age'].mean()
	print("avg_patient_age =", avg_patient_age, "\n")
	
	age_by_grade = patient_data.groupby('who')['age'].mean()
	print("age_by_grade")
	print(age_by_grade, "\n")

	preop_sz_by_grade = patient_data.groupby('who')['preop_sz'].value_counts().unstack()
	print(preop_sz_by_grade, "\n")
	
	late_sz_by_grade = patient_data.groupby('who')['late_sz'].value_counts().unstack()
	print(late_sz_by_grade, "\n")
	
	preop_sz_by_idh = patient_data.groupby('idh')['preop_sz'].value_counts().unstack()
	print(preop_sz_by_idh, "\n")
	
	idh_by_grade = patient_data.groupby('who')['idh'].value_counts().unstack()
	print (idh_by_grade, "\n")

	late_sz_by_hyperexcitability = patient_data.groupby('hyperexcitability')['late_sz'].value_counts().unstack()
	print(late_sz_by_hyperexcitability, "\n")
	chi2, p, dof, expected = chi2_contingency(late_sz_by_hyperexcitability, correction=False)
	print("chi2 p-value:", p)

	ever_sz_by_hyperexcitability = patient_data.groupby('hyperexcitability')['ever_sz'].value_counts().unstack()
	print(ever_sz_by_hyperexcitability, "\n")
	chi2, p, dof, expected = chi2_contingency(ever_sz_by_hyperexcitability, correction=False)
	print("chi2 p-value:", p)

	patient_data.to_csv("~/Desktop/patient_data.csv",encoding="utf-8-sig")
	eeg_data.to_csv("~/Desktop/eeg_data.csv", encoding="utf-8-sig")

	print(patient_data.dtypes)

	x = patient_data[['hyperexcitability', 'idh']]
	x = sm.add_constant(x)
	y = patient_data['ever_sz']

	model = sm.Logit(y,x).fit()
	print(model.summary())


	params = model.params
	conf = model.conf_int()
	odds_ratios = np.exp(params)
	odds_ci = np.exp(conf)

	print(pd.DataFrame({
		"OR": odds_ratros,
		"2.5%": odds_ci[0],
		"97.5%": odds_ci[1]
	}))




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

	return (multchoice_dict, patient_data, eeg_data)

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
	lpd = eeg_data.loc[eeg_data['lpd_yn'] == "1", 'study_id'].drop_duplicates()
	bipd = eeg_data.loc[eeg_data['bipd_yn'] == "1", 'study_id'].drop_duplicates()
	lrda = eeg_data.loc[eeg_data['lrda_yn'] == "1", 'study_id'].drop_duplicates()
	sz = eeg_data.loc[eeg_data['sz_yn'] == "1", 'study_id'].drop_duplicates()
	ied = eeg_data.loc[eeg_data['epi_disch_yn'] == "1", 'study_id'].drop_duplicates()

	hyperexcitable = pd.concat([gpd, lpd, bipd, lrda, sz, ied], ignore_index=True)

	patient_data['hyperexcitability'] = patient_data['study_id'].isin(hyperexcitable)

	return (patient_data)

# determines if patient ever had a clinical seizure
def find_ever_sz (patient_data):
	patient_data = patient_data.copy()

	ever_sz = patient_data.loc[(patient_data['preop_sz'] == "Yes") | (patient_data['late_sz'] == "Yes")]
	print (ever_sz)

	patient_data['ever_sz'] = patient_data['study_id'].isin(ever_sz['study_id'])

	return (patient_data)

if __name__ == "__main__":
	import sys
	retry_stats (str(sys.argv[1]), str(sys.argv[2]))
