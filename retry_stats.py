#! /usr/bin/env python

import os
import pandas as pd
from retry_parser import redcap_parser
from retry_models import logistic_regression, ordinal_regression
from retry_plots import plot_descriptive_stats, plot_forest_plot

def retry_stats (data_file, dict_file):

	output_path = os.path.dirname(data_file)
	
	# Parse data and dictionary files and return patient and eeg dataframes
	patient_data, eeg_data = redcap_parser(data_file, dict_file)

	# Parse dataframes to find high-level patient characteristics
	patient_data = find_hyperexcitability (patient_data, eeg_data)
	patient_data = find_ever_sz (patient_data)

	# Output individual dataframes (if needed for debugging)
	patient_data.to_csv(os.path.join(output_path, "patient_data.csv"),encoding="utf-8-sig", index=False, na_rep="NaN")
	eeg_data.to_csv(os.path.join(output_path, "eeg_data.csv"), encoding="utf-8-sig", index=False, na_rep="NaN")

	# Descriptive statistics
	plot_descriptive_stats (patient_data, output_path)

	# Regression models
	model_name = "ever hyperexcitability (esz, ied, rpp) vs. ever seizures"
	model = logistic_regression (patient_data, 'ever_sz', ['hyperexcitability', 'idh', 'who', 'age'], model_name, output_path)
	plot_forest_plot(model, model_name, output_path)

	model_name = "ever specific EEG factors (rpp, esz, ied) vs. ever seizures"
	model = logistic_regression (patient_data, 'ever_sz', ['rpp', 'esz', 'ied', 'idh', 'who', 'age'], model_name, output_path)
	plot_forest_plot(model, model_name, output_path)

	model_name = "ever specific EEG factors (any_disc, lrda, esz) vs. ever seizures"
	model = logistic_regression (patient_data, 'ever_sz', ['any_disc', 'lrda', 'esz', 'idh', 'who', 'age'], model_name, output_path)
	plot_forest_plot(model, model_name, output_path)

	model_name = "ever specific EEG factors (any_disc, lrda) vs. ever seizures"
	model = logistic_regression (patient_data, 'ever_sz', ['any_disc', 'lrda', 'idh', 'who', 'age'], model_name, output_path)
	plot_forest_plot(model, model_name, output_path)

	model_name = "ever specific EEG factors (pd, ied, lrda) vs. ever seizures"
	model = logistic_regression (patient_data, 'ever_sz', ['pd', 'ied', 'lrda', 'idh', 'who', 'age'], model_name, output_path)
	plot_forest_plot(model, model_name, output_path)

	model_name = "ever hyperexcitability vs. late seizures"
	pts = (patient_data['preop_sz'] == False) | patient_data['preop_sz'].isna()
	patients_wo_baseline_sz = patient_data[pts].copy()
	model = logistic_regression (patients_wo_baseline_sz, 'late_sz', ['hyperexcitability', 'idh', 'who', 'age'], model_name, output_path)
	plot_forest_plot(model, model_name, output_path)

	model_name = "ever specific EEG factors (any_disc, lrda, esz) vs. late seizures"
	pts = (patient_data['preop_sz'] == False) | patient_data['preop_sz'].isna()
	patients_wo_baseline_sz = patient_data[pts].copy()
	model = logistic_regression (patients_wo_baseline_sz, 'late_sz', ['any_disc', 'lrda', 'esz', 'idh', 'who', 'age'], model_name, output_path)
	plot_forest_plot(model, model_name, output_path)

	model_name = "ever specific EEG factors (any_disc, lrda) vs. late seizures"
	pts = (patient_data['preop_sz'] == False) | patient_data['preop_sz'].isna()
	patients_wo_baseline_sz = patient_data[pts].copy()
	model = logistic_regression (patients_wo_baseline_sz, 'late_sz', ['any_disc', 'lrda', 'idh', 'who', 'age'], model_name, output_path)
	plot_forest_plot(model, model_name, output_path)

	model_name = "ever hyperexcitability vs. grade"
	model = ordinal_regression (patient_data, 'who', ['hyperexcitability', 'idh', 'age'], model_name, output_path)
	plot_forest_plot(model, model_name, output_path)

	model_name = "specific EEG factors (lpd, gpd, lrda, esz, ied) vs. grade"
	model = ordinal_regression (patient_data, 'who', ['lpd', 'gpd', 'lrda', 'esz', 'ied', 'idh', 'age'], model_name, output_path)
	plot_forest_plot(model, model_name, output_path)

	model_name = "hemorrhage, necrosis vs. ever_sz"
	model = ordinal_regression (patient_data, 'ever_sz', ['hemorrhage', 'necrosis', 'idh', 'who', 'age'], model_name, output_path)
	plot_forest_plot(model, model_name, output_path)

	model_name = "hemorrhage, necrosis vs. pd"
	model = ordinal_regression (patient_data, 'pd', ['hemorrhage', 'necrosis', 'idh', 'who', 'age'], model_name, output_path)
	plot_forest_plot(model, model_name, output_path)

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

if __name__ == "__main__":
	import sys
	retry_stats (str(sys.argv[1]), str(sys.argv[2]))
