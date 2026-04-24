import pandas as pd
import numpy as np

def redcap_parser (data_file, dict_file):

	# Import CSV files as DataFrames
	raw_data = pd.read_csv(data_file, dtype=str)	# Raw patient and EEG data exported as CSV from RedCap
	data_dict = pd.read_csv(dict_file, dtype=str)	# Data Dictionary exported as CSV file from RedCap

	# Create separate dataframes of radio/dropdown and checkbox Data Dictionary entries
	multchoice_dict = data_dict[(data_dict['Field Type'] == 'radio')|(data_dict['Field Type'] == 'dropdown')].copy()
	checkbox_dict = data_dict[(data_dict['Field Type'] == 'checkbox')].copy()

	# Convert df column to an actual Python dictionary and append it as another column
	multchoice_dict.loc[:,'parsed_dict'] = multchoice_dict.loc[:,'Choices, Calculations, OR Slider Labels'].apply(convert_to_dict)
	checkbox_dict.loc[:,'parsed_dict'] = checkbox_dict.loc[:,'Choices, Calculations, OR Slider Labels'].apply(convert_to_dict)
	
	# In df containing patient/EEG data, replace meaningless numbers with their meaning using dictionary
	multchoice_cleaned_data = replace_with_dict(raw_data, multchoice_dict)
	cleaned_data = replace_with_dict_checkboxes(multchoice_cleaned_data, checkbox_dict)

	# create a dataframe of just the patient data
	patient_data = cleaned_data[cleaned_data['redcap_repeat_instrument'] != 'eeg_data'].copy()
	patient_data = patient_data.loc[:,'study_id':'glioma_complete'].copy()

	# exclude patients with WHO grade 1 glioma
	patient_data = patient_data[patient_data['who'] != '1'].copy()

	# create a dataframe of just the EEG data
	eeg_data = cleaned_data[cleaned_data['redcap_repeat_instrument'] == 'eeg_data'].copy()
	eeg_data_1 = eeg_data.loc[:,'study_id':'redcap_repeat_instance'].copy()
	eeg_data_2 = eeg_data.loc[:,'eeg_inclusion':'eeg_data_complete'].copy()
	eeg_data = pd.concat([eeg_data_1,eeg_data_2], axis=1)

	# remove excluded patients
	patient_data = patient_data[patient_data['inclusion'].str.strip() == 'INCLUDED'].copy()
	included_study_ids = patient_data['study_id'].unique()
	eeg_data = eeg_data[eeg_data['study_id'].isin(included_study_ids)].copy()

	# remove excluded EEGs
	eeg_data = eeg_data[eeg_data['eeg_inclusion'].str.strip() == 'INCLUDED'].copy()

	# clean up data types in patient data
	patient_data["age"] = patient_data["age"].astype(float)
	patient_data["study_id"] = patient_data["study_id"].astype(int)
	patient_data['idh'] = patient_data['idh'].map({'Yes': True, 'No': False, '': pd.NA, np.nan: pd.NA}).astype("boolean")
	patient_data['codel'] = patient_data['codel'].map({'Yes': True, 'No': False, '': pd.NA, np.nan: pd.NA}).astype("boolean")
	patient_data['mgmt'] = patient_data['mgmt'].map({'Yes': True, 'No': False, '': pd.NA, np.nan: pd.NA}).astype("boolean")
	patient_data['preop_sz'] = patient_data['preop_sz'].map({'Yes': True, 'No': False, '': pd.NA, np.nan: pd.NA}).astype("boolean")
	patient_data['late_sz'] = patient_data['late_sz'].map({'Yes': True, 'No': False, '': pd.NA, np.nan: pd.NA}).astype("boolean")
	patient_data['hemorrhage'] = patient_data['hemorrhage'].map({'Yes': True, 'No': False, '': pd.NA, np.nan: pd.NA}).astype("boolean")
	patient_data['necrosis'] = patient_data['necrosis'].map({'Yes': True, 'No': False, '': pd.NA, np.nan: pd.NA}).astype("boolean")
	patient_data['who'] = patient_data['who'].map({'1': 1, '2': 2, '3': 3, '4': 4, 'Indeterminate': np.nan})

	# clean up data types in EEG data
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
		base_name = row['Variable / Field Name']
		replacement_dict = row['parsed_dict']

		for code, label in replacement_dict.items():
			col_name = f"{base_name}___{code}"
			if col_name in raw_data.columns:
				raw_data.rename(columns={col_name: f"{base_name}_{label}"}, inplace=True)

		# if column_name in raw_data.columns:
		# 	result = replacement_dict.get(column_name)
		# 	raw_data.rename(columns={column_name:result}, inplace=True)

			# output = []
			# for i, this_item in raw_data[column_name].items():
			# 	this_string =str(this_item)
			# 	result = ",".join(replacement_dict.get(item.strip(), item.strip()) for item in this_string.split(","))
			# 	output.append(result)
			# raw_data[column_name] = output

	return (raw_data)

def find_hyperexcitability (patient_data, eeg_data):
	patient_data = patient_data.copy()
	eeg_data = eeg_data.copy()

	gpd = eeg_data.loc[eeg_data['gpd_yn'] == "1", 'study_id'].drop_duplicates()
	lpd = eeg_data.loc[eeg_data['lpd_yn'] == "1", 'study_id'].drop_duplicates()
	bipd = eeg_data.loc[eeg_data['bipd_yn'] == "1", 'study_id'].drop_duplicates()
	lrda = eeg_data.loc[eeg_data['lrda_yn'] == "1", 'study_id'].drop_duplicates()
	esz = eeg_data.loc[eeg_data['sz_yn'] == "1", 'study_id'].drop_duplicates()
	ied = eeg_data.loc[eeg_data['epi_disch_yn'] == "1", 'study_id'].drop_duplicates()

	new_cols = pd.DataFrame({
		'gpd': patient_data['study_id'].isin(gpd).astype("boolean"),
		'lpd': patient_data['study_id'].isin(lpd).astype("boolean"),
		'bipd': patient_data['study_id'].isin(bipd).astype("boolean"),
		'lrda': patient_data['study_id'].isin(lrda).astype("boolean"),
		'esz': patient_data['study_id'].isin(esz).astype("boolean"),
		'ied': patient_data['study_id'].isin(ied).astype("boolean"),
		'rpp': patient_data['study_id'].isin(pd.concat([gpd, lpd, bipd, lrda], ignore_index=True)).astype("boolean"),
		'any_disc': patient_data['study_id'].isin(pd.concat([gpd, lpd, bipd, ied], ignore_index=True)).astype("boolean"),
		'pd': patient_data['study_id'].isin(pd.concat([gpd, lpd, bipd], ignore_index=True)).astype("boolean"),
		'hyperexcitability': patient_data['study_id'].isin(pd.concat([gpd, lpd, bipd, lrda, esz, ied], ignore_index=True)).astype("boolean"),
	}, index=patient_data.index)

	patient_data = pd.concat([patient_data, new_cols], axis=1)

	return (patient_data)

# determines if patient ever had a clinical seizure
def find_ever_sz (patient_data):
	patient_data = patient_data.copy()

	ever_sz = patient_data.loc[(patient_data['preop_sz'] == True) | (patient_data['late_sz'] == True)]

	patient_data['ever_sz'] = patient_data['study_id'].isin(ever_sz['study_id'])

	return (patient_data)