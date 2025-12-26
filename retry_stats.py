#! /usr/bin/env python

import pandas as pd

def retry_stats (data_file, dict_file):
	
	multchoice_dict, patient_data, eeg_data = redcap_parser(data_file, dict_file)
	
	
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


def redcap_parser (data_file, dict_file):
	
	# Import CSV files as DataFrames
	raw_data = pd.read_csv(data_file, dtype=str)	# raw patient and EEG data exported as CSV from RedCap
	data_dict = pd.read_csv(dict_file)					# Data Dictionary exported as CSV file from RedCap

	# Remove instruments not included in this dataset from Data Dictionary df (reduces number of error messages)
	data_dict = data_dict[data_dict['Form Name'] != 'primary_brain_tumors']
	# data_dict = data_dict[data_dict['Form Name'] != 'noc_tracking']
	# data_dict = data_dict[data_dict['Form Name'] != 'ranotreat']

	# Create separate dataframes of radio/dropdown and checkbox Data Dictionary entries
	multchoice_dict = data_dict[(data_dict['Field Type'] == 'radio')|(data_dict['Field Type'] == 'dropdown')]
	checkbox_dict = data_dict[(data_dict['Field Type'] == 'checkbox')]

	# Convert df column to an actual Python dictionary and append it as another column
	multchoice_dict.loc[:,'parsed_dict'] = multchoice_dict.loc[:,'Choices, Calculations, OR Slider Labels'].apply(convert_to_dict)
	checkbox_dict.loc[:,'parsed_dict'] = checkbox_dict.loc[:,'Choices, Calculations, OR Slider Labels'].apply(convert_to_dict)
	
	# In df containing patient/EEG data, replace meaningless numbers with their meaning using dictionary
	cleaned1_data = replace_with_dict(raw_data, multchoice_dict)
	checkbox_dict.to_csv("~/Desktop/checkbox_dict.csv")
	cleaned_data = replace_with_dict_checkboxes(cleaned1_data, checkbox_dict)
	cleaned_data.to_csv("~/Desktop/cleaned_data.csv")



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

def replace_with_dict(df, multchoice_dict):
	for index, row in multchoice_dict.iterrows():
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

def replace_with_dict_checkboxes(df, checkbox_dict):
	new_df = df
	for index, row in checkbox_dict.iterrows():
		var_name = row['Variable / Field Name']
		replacement_dict = row['parsed_dict']
		
		for (col, _) in df.items():
			if var_name in col:
				[this_name, this_number] = df[col].name.split("___")
				print(this_number)
				print(this_name)
				this_meaning = replacement_dict[this_number]
				new_name = this_name + "___" + this_meaning
				print(new_name)
				new_df.rename(columns={col:new_name}, inplace=False)

	return new_df
	

if __name__ == "__main__":
	import sys
	retry_stats (str(sys.argv[1]), str(sys.argv[2]))
