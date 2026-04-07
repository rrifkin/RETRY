import os
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

def plot_descriptive_stats (patient_data, output_dir):
	output_dir = os.path.expanduser(output_dir)
	fig, axes = plt.subplots(2,2,figsize=(12,10))

	# preop sz by WHO grade
	preop_sz_by_grade_data = patient_data.groupby('who')['preop_sz'].value_counts(normalize=True).unstack().fillna(0) * 100
	preop_sz_by_grade_data = preop_sz_by_grade_data.reindex(columns=[True, False], fill_value=0)
	preop_sz_by_grade_data.plot(kind = 'bar', stacked=True, ax = axes[0,0], color=['coral', 'steelblue'], edgecolor='black')
	axes[0, 0].set_title("Pre-operative Seizures by WHO Grade")
	axes[0, 0].set_xlabel("WHO Grade")
	axes[0, 0].set_ylabel("Percentage")
	axes[0, 0].legend(title="Pre-op Seizure", labels=["Yes", "No"])
	axes[0, 0].tick_params(axis='x', rotation=0)

	# late sz by WHO grade
	late_sz_by_grade_data = patient_data.groupby('who')['late_sz'].value_counts(normalize=True).unstack().fillna(0) * 100
	late_sz_by_grade_data = late_sz_by_grade_data.reindex(columns=[True, False], fill_value=0)
	late_sz_by_grade_data.plot(kind = 'bar', stacked=True, ax = axes[0,1], color=['coral', 'steelblue'], edgecolor='black')
	axes[0, 1].set_title("Late Seizures by WHO Grade")
	axes[0, 1].set_xlabel("WHO Grade")
	axes[0, 1].set_ylabel("Percentage")
	axes[0, 1].legend(title="Late Seizure", labels=["Yes", "No"])
	axes[0, 1].tick_params(axis='x', rotation=0)

	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "descriptive_stats.png"), dpi=300)
	plt.close()
	print("Saved descriptive_stats.png to Desktop")