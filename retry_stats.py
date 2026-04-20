#! /usr/bin/env python

import os
import pandas as pd
from retry_parser import redcap_parser
from retry_models import logistic_regression, ordinal_regression
from retry_plots import plot_descriptive_stats, plot_forest_plot

def retry_stats (data_file, dict_file):

	data_file_path = os.path.dirname(data_file)
	
	# Parse data and dictionary files and return patient and eeg dataframes
	patient_data, eeg_data = redcap_parser(data_file, dict_file)

	# Parse dataframes to find high-level patient characteristics
	patient_data = find_hyperexcitability (patient_data, eeg_data)
	patient_data = find_ever_sz (patient_data)

	patient_data.to_csv(os.path.join(data_file_path, "patient_data.png"),encoding="utf-8-sig", index=False, na_rep="NaN")
	eeg_data.to_csv(os.path.join(data_file_path, "eeg_data.png"), encoding="utf-8-sig", index=False, na_rep="NaN")
	descriptive_statistics (patient_data)
	plot_descriptive_stats (patient_data, data_file_path)

	# hyperexcitability <-> ever having seizures
	print("\nIs there an association between EEG hyperexcitability (esz, ied, rpp) at any time point, and clinical seizure at any time point?")
	ever_sz_hyperexcitability = logistic_regression (patient_data, 'ever_sz', ['hyperexcitability', 'idh', 'who', 'age'])
	plot_forest_plot(ever_sz_hyperexcitability, "ever_sz_hyperexcitability", data_file_path)

	# specific EEG factors (rpp, esz, ied) <-> ever having seizures
	print("\nIs there an association between specific EEG factors (esz, ied, rpp) at any time point, and clinical seizure at any time point?")
	ever_sz_specific = logistic_regression (patient_data, 'ever_sz', ['rpp', 'esz', 'ied', 'idh', 'who', 'age'])
	plot_forest_plot(ever_sz_specific, "ever_sz_specific", data_file_path)

	# specific EEG factors (any_disc, lrda, esz) <-> ever having seizures
	print("\nIs there an association between specific EEG factors (any_disc, lrda, esz) at any time point, and clinical seizure at any time point?")
	ever_sz_specific2 = logistic_regression (patient_data, 'ever_sz', ['any_disc', 'lrda', 'esz', 'idh', 'who', 'age'])
	plot_forest_plot(ever_sz_specific2, "ever_sz_specific2", data_file_path)

	# hyperexcitability <-> late-onset seizures
	print("\nIs there an association between EEG hyperexcitability (esz, ied, rpp) at any time point, and late-onset clinical seizures?")
	pts = (patient_data['preop_sz'] == False) | patient_data['preop_sz'].isna()
	patients_wo_baseline_sz = patient_data[pts].copy()
	late_sz_hyperexcitability = logistic_regression (patients_wo_baseline_sz, 'late_sz', ['hyperexcitability', 'idh', 'who', 'age'])
	plot_forest_plot(late_sz_hyperexcitability, "late_sz_hyperexcitability", data_file_path)

	# specific EEG factors <-> late-onset seizures
	print("\nIs there an association between specific EEG factors (any_disc, lrda, esz) at any time point, and late-onset clinical seizures?")
	pts = (patient_data['preop_sz'] == False) | patient_data['preop_sz'].isna()
	patients_wo_baseline_sz = patient_data[pts].copy()
	late_sz_specific = logistic_regression (patients_wo_baseline_sz, 'late_sz', ['any_disc', 'lrda', 'esz', 'idh', 'who', 'age'])
	plot_forest_plot(late_sz_specific, "late_sz_specific", data_file_path)

	# hyperexcitability <-> tumor grade
	print("\nIs there an association between EEG hyperexcitability (esz, ied, rpp) and WHO tumor grade?\n")
	grade_hyperexcitability = ordinal_regression (patient_data, 'who', ['hyperexcitability', 'idh', 'age'])
	plot_forest_plot(grade_hyperexcitability, "grade_hyperexcitability", data_file_path)

	# specific EEG factors <-> tumor grade
	print("\nIs there an association between specific EEG factors (lpd, gpd, lrda, esz, ied) and WHO tumor grade?\n")
	grade_specific = ordinal_regression (patient_data, 'who', ['lpd', 'gpd', 'lrda', 'esz', 'ied', 'idh', 'age'])
	plot_forest_plot(grade_specific, "grade_specific", data_file_path)

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

def descriptive_statistics (patient_data):
	# Descriptive statistics
	num_included = len(patient_data)	
	print("Number of patients included (n) =", num_included, "\n")

	avg_patient_age = patient_data['age'].mean()
	print("Average patient age =", avg_patient_age, "\n")

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

if __name__ == "__main__":
	import sys
	retry_stats (str(sys.argv[1]), str(sys.argv[2]))
