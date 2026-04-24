#! /usr/bin/env python
import os
import pandas as pd
from retry_parser import redcap_parser, find_hyperexcitability, find_ever_sz
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
	model_name = "location vs. ever seizures"
	model = logistic_regression (patient_data, 'ever_sz', ['loc_Frontal', 'loc_Parietal','loc_Occipital','loc_Temporal','loc_Subcortical','loc_Insula','loc_Infratentorial','idh', 'who', 'age'], model_name, output_path)
	plot_forest_plot(model, model_name, output_path)

	model_name = "complications vs. ever seizures"
	complication_cols = [col for col in patient_data.columns if col.startswith('complications_')]
	complication_cols = [col for col in complication_cols if patient_data[col].nunique() > 1]
	model = logistic_regression (patient_data, 'ever_sz', complication_cols + ['idh', 'who', 'age'], model_name, output_path)
	plot_forest_plot(model, model_name, output_path)

	model_name = "ever hyperexcitability vs. ever seizures"
	model = logistic_regression (patient_data, 'ever_sz', ['hyperexcitability', 'idh', 'who', 'age'], model_name, output_path)
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

	# model_name = "mutations vs. ever seizures"
	# mutation_cols = [col for col in patient_data.columns if col.startswith('mutations_')]
	# mutation_cols = [col for col in mutation_cols if patient_data[col].nunique() > 1]
	# mutation_df = patient_data[mutation_cols]
	# mutation_df = mutation_df.loc[:, ~mutation_df.T.duplicated()]
	# mutation_cols = mutation_df.columns.tolist()
	# model = logistic_regression (patient_data, 'ever_sz', mutation_cols + ['idh', 'who', 'age'], model_name, output_path)
	# plot_forest_plot(model, model_name, output_path)

	# model_name = "ever specific EEG factors (rpp, esz, ied) vs. ever seizures"
	# model = logistic_regression (patient_data, 'ever_sz', ['rpp', 'esz', 'ied', 'idh', 'who', 'age'], model_name, output_path)
	# plot_forest_plot(model, model_name, output_path)

	# model_name = "ever specific EEG factors (any_disc, lrda, esz) vs. ever seizures"
	# model = logistic_regression (patient_data, 'ever_sz', ['any_disc', 'lrda', 'esz', 'idh', 'who', 'age'], model_name, output_path)
	# plot_forest_plot(model, model_name, output_path)

	# model_name = "ever specific EEG factors (any_disc, lrda, esz) vs. late seizures"
	# pts = (patient_data['preop_sz'] == False) | patient_data['preop_sz'].isna()
	# patients_wo_baseline_sz = patient_data[pts].copy()
	# model = logistic_regression (patients_wo_baseline_sz, 'late_sz', ['any_disc', 'lrda', 'esz', 'idh', 'who', 'age'], model_name, output_path)
	# plot_forest_plot(model, model_name, output_path)

if __name__ == "__main__":
	import sys
	retry_stats (str(sys.argv[1]), str(sys.argv[2]))
