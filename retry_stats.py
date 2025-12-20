#! /usr/bin/env python

import pandas as pd

def retry_stats (data_file, dict):
	
	multchoice_dict, patient_data, eeg_data = redcap_parser(data_file, dict)
	
	clean_patient_data = replace_with_dict(patient_data, multchoice_dict)
	
	# avg_patient_age = patient_data['age'].mean()
	#print("avg_patient_age")
	#print(avg_patient_age)
	
	# age_by_grade = clean_patient_data.groupby('who')['age'].mean()
	#print(age_by_grade)
	
	late_sz_by_grade = patient_data.groupby('who')['any_sz'].value_counts()
	print(late_sz_by_grade)
	
	preop_sz_by_grade = patient_data.groupby('who')['preop_sz'].value_counts()
	print(preop_sz_by_grade)
	
	preop_sz_by_idh = patient_data.groupby('idh')['preop_sz'].value_counts()
	print(preop_sz_by_idh)
	
	idh_by_grade = patient_data.groupby('who')['idh'].value_counts()
	print (idh_by_grade)


def redcap_parser (data_file, dict):
	raw_data = pd.read_csv(data_file, dtype=str)
	dict = pd.read_csv(dict)

	# create a dictionary to convert numeric answers to their meanings
	dict = dict[dict['Form Name'] != 'primary_brain_tumors']
	dict = dict[dict['Form Name'] != 'noc_tracking']
	dict = dict[dict['Form Name'] != 'ranotreat']

	multchoice_dict = dict[(dict['Field Type'] == 'radio')|(dict['Field Type'] == 'dropdown')|(dict['Field Type'] == 'checkbox') ]
	multchoice_dict.loc[:,'parsed_dict'] = multchoice_dict.loc[:,'Choices, Calculations, OR Slider Labels'].apply(convert_to_dict)
	multchoice_dict = multchoice_dict.loc[:,['Variable / Field Name', 'parsed_dict']]
	
	# create a dataframe of just the patient data
	patient_data = raw_data[raw_data['redcap_repeat_instrument'] != 'eeg_data']
	patient_data = patient_data.loc[:,'study_id':'glioma_complete']

	# create a dataframe of just the EEG data
	eeg_data = raw_data[raw_data['redcap_repeat_instrument'] == 'eeg_data']
	eeg_data_1 = eeg_data.loc[:,'study_id':'redcap_repeat_instance']
	eeg_data_2 = eeg_data.loc[:,'eeg_inclusion':'eeg_data_complete']
	eeg_data = pd.concat([eeg_data_1,eeg_data_2], axis=1)
	
	return (multchoice_dict, patient_data, eeg_data)

def convert_to_dict (string):
	return {pair.split(', ')[0]: pair.split(', ')[1].strip() for pair in string.split(' | ')}

def replace_with_dict(df, dict):
	for index, row in dict.iterrows():
		column_name = row['Variable / Field Name']
		replacement_dict = row['parsed_dict']
		
		try:
			if column_name in df.columns:
				df[column_name] = df[column_name].replace(replacement_dict)
			else:
				#pass				
				print(column_name, " does not exist in the DataFrame.")
		except KeyError as e:
			#pass
			print("KeyError ", e)
	return df
	

if __name__ == "__main__":
	import sys
	retry_stats (str(sys.argv[1]), str(sys.argv[2]))