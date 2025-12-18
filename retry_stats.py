#! /usr/bin/env python

import pandas as pd

def redcap_parser (data_file, dict):

	raw_data = pd.read_csv(data_file)
	dict = pd.read_csv(dict)

	# create a dictionary to convert numeric answers to their meanings
	multchoice_dict = dict[(dict['Field Type'] == 'radio')|(dict['Field Type'] == 'dropdown')|(dict['Field Type'] == 'checkbox') ]
	multchoice_dict.loc[:,'parsed_dict'] = multchoice_dict.loc[:,'Choices, Calculations, OR Slider Labels'].apply(convert_to_dict)
	multchoice_dict = multchoice_dict.loc[:,['Variable / Field Name', 'parsed_dict']]
	print(multchoice_dict)
	
	# create a dataframe of just the patient data
	patient_data = raw_data[raw_data['redcap_repeat_instrument'] != 'eeg_data']
	patient_data = patient_data.loc[:,'study_id':'glioma_complete']
	#print(patient_data)

	# create a dataframe of just the EEG data
	eeg_data = raw_data[raw_data['redcap_repeat_instrument'] == 'eeg_data']
	eeg_data_1 = eeg_data.loc[:,'study_id':'redcap_repeat_instance']
	eeg_data_2 = eeg_data.loc[:,'eeg_inclusion':'eeg_data_complete']
	eeg_data = pd.concat([eeg_data_1,eeg_data_2], axis=1)
	#print(eeg_data)
	
	return (multchoice_dict, patient_data, eeg_data)

	
	avg_patient_age = patient_data['age'].mean()

	age_by_grade = patient_data.groupby('who')['age'].mean()

	
	sz_by_grade = patient_data.groupby('who')['any_sz'].value_counts()

def convert_to_dict (string):
	return {pair.split(', ')[0]: pair.split(', ')[1].strip() for pair in string.split(' | ')}

# def retry_stats (file):

	
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