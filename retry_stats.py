#! /usr/bin/env python

import pandas as pd


def redcap_parser (data_file, data_dictionary):

	raw_data = pd.read_csv(data_file)
	data_dictionary = pd.read_csv(data_dictionary)
	
	data_dictionary = data_dictionary[data_dictionary['Form Name'] != 'primary_brain_tumors']
	data_dictionary = data_dictionary.loc[:,['Variable / Field Name', 'Choices, Calculations, OR Slider Labels']]
	print(data_dictionary)

	data_dictionary = data_dictionary.loc[:,'Choices, Calculations, OR Slider Labels'].str.replace(',' , ':' , regex=False)
	data_dictionary = data_dictionary.loc[:,'Choices, Calculations, OR Slider Labels'].str.replace('|' , ',' , regex=False)

	print(data_dictionary)
	
	eeg_data = raw_data[raw_data['redcap_repeat_instrument'] == 'eeg_data']
	eeg_data_1 = eeg_data.loc[:,'study_id':'redcap_repeat_instance']
	eeg_data_2 = eeg_data.loc[:,'eeg_inclusion':'eeg_data_complete']
	eeg_data = pd.concat([eeg_data_1,eeg_data_2], axis=1)
	
	patient_data = raw_data[raw_data['redcap_repeat_instrument'] != 'eeg_data']
	patient_data = patient_data.loc[:,'study_id':'glioma_complete']
	

	
	avg_patient_age = patient_data['age'].mean()

	age_by_grade = patient_data.groupby('who')['age'].mean()

	
	sz_by_grade = patient_data.groupby('who')['any_sz'].value_counts()


# def retry_stats (file):
# 	raw_data = pd.read_csv(file)
# 	num_entries = raw_data['study_id'].nunique()
# 	print("number of patients = ", num_entries)

# 	eeg_data = raw_data[raw_data['redcap_repeat_instrument'] == 'eeg_data']
# 	eeg_data_1 = eeg_data.loc[:,'study_id':'redcap_repeat_instance']
# 	eeg_data_2 = eeg_data.loc[:,'eeg_inclusion':'eeg_data_complete']
# 	eeg_data = pd.concat([eeg_data_1,eeg_data_2], axis=1)
	
# 	patient_data = raw_data[raw_data['redcap_repeat_instrument'] != 'eeg_data']
# 	patient_data = patient_data.loc[:,'study_id':'glioma_complete']
	
# 	print("eeg_data")
# 	print(eeg_data.head())
	
# 	print("patient_data")
# 	print(patient_data.head())
	
# 	avg_patient_age = patient_data['age'].mean()
# 	print("avg_patient_age")
# 	print(avg_patient_age)
	
# 	age_by_grade = patient_data.groupby('who')['age'].mean()
# 	print(age_by_grade)
	
# 	sz_by_grade = patient_data.groupby('who')['any_sz'].value_counts()
# 	print(sz_by_grade)

if __name__ == "__main__":
	import sys
	redcap_parser (str(sys.argv[1]), str(sys.argv[2]))