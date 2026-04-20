import os
import numpy as np
import matplotlib.pyplot as plt

def plot_forest_plot(model_results, model_name, output_dir):
	"""
	model_results: dict with keys as variable names, values as (coef, ci_lower, ci_upper)
	model_name: string for the plot title
	output_dir: path to save the plot
	"""
	output_dir = os.path.expanduser(output_dir)

	variables = list(model_results.keys())
	coefs = [model_results[v][0] for v in variables]
	ci_lowers = [model_results[v][1] for v in variables]
	ci_uppers = [model_results[v][2] for v in variables]

	# Calculate error bars (distance from coef to CI bounds)
	errors = [
		[coefs[i] - ci_lowers[i] for i in range(len(coefs))],
		[ci_uppers[i] - coefs[i] for i in range(len(coefs))]
	]

	fig, ax = plt.subplots(figsize=(10, 0.5 * len(variables) + 1))

	y_pos = np.arange(len(variables))
	ax.errorbar(coefs, y_pos, xerr=errors, fmt='o', markersize=8, capsize=5, capthick=2, color='steelblue')
	ax.axvline(x=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
	ax.set_xlim(0,10)
	ax.set_ylim(-0.5, len(variables) - 0.5)

	ax.set_yticks(y_pos)
	ax.set_yticklabels(variables)
	ax.set_xlabel("Odds Ratio (95% CI)")
	ax.set_title(f"Forest Plot: {model_name}")
	ax.grid(axis='x', alpha=0.3)

	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, f"forest_{model_name}.png"), dpi=300)
	plt.close()
	print(f"Saved forest_{model_name}.png to {output_dir}")

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
	preop_sz_n = patient_data.dropna(subset=['preop_sz']).groupby('who').size()
	annotate_group_n (axes[0, 0], preop_sz_by_grade_data, preop_sz_n)

	# late sz by WHO grade
	late_sz_by_grade_data = patient_data.groupby('who')['late_sz'].value_counts(normalize=True).unstack().fillna(0) * 100
	late_sz_by_grade_data = late_sz_by_grade_data.reindex(columns=[True, False], fill_value=0)
	late_sz_by_grade_data.plot(kind = 'bar', stacked=True, ax = axes[0,1], color=['coral', 'steelblue'], edgecolor='black')
	axes[0, 1].set_title("Late Seizures by WHO Grade")
	axes[0, 1].set_xlabel("WHO Grade")
	axes[0, 1].set_ylabel("Percentage")
	axes[0, 1].legend(title="Late Seizure", labels=["Yes", "No"])
	axes[0, 1].tick_params(axis='x', rotation=0)
	late_sz_n = patient_data.dropna(subset=['late_sz']).groupby('who').size()
	annotate_group_n (axes[0, 1], late_sz_by_grade_data, late_sz_n)

	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "descriptive_stats.png"), dpi=300)
	plt.close()
	print("Saved descriptive_stats.png to Desktop")

def annotate_group_n(ax, pct_df, n_by_group):
	# pct_df index order matches bar order
	for i, group in enumerate(pct_df.index):
		n_val = int(n_by_group.get(group, 0))
		ax.text(
			i, 2, f"n={n_val}",
			ha="center", va="bottom",
			fontsize=9, fontweight="bold", color="white",
			bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.35, edgecolor="none")
		)
	# ax.set_ylim(0, max(110, float(pct_df.sum(axis=1).max()) + 10))