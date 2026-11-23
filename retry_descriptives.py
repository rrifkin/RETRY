import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LOBES = ['Frontal', 'Temporal', 'Insula', 'Parietal', 'Occipital', 'Subcortical', 'Infratentorial']

# EEG duration categories, shortest to longest (same order as the REDCap data dictionary)
DURATION_ORDER = ['< 40 minutes', '≥ 40 minutes', '≥ 60 minutes', '≥ 2 hours', '≥ 12 hours',
				  '≥ 26 and < 36 hours', '≥36 hours', '≥ 60 hours', '≥ 84 hours']

# Checkbox columns come out of the parser as the strings '0' / '1'
def checked (df, col):
	return pd.to_numeric(df[col], errors='coerce').fillna(0).eq(1)

def pct (n, total):
	return (n / total * 100) if total > 0 else float('nan')

def retry_descriptives (patient_data, eeg_data, output_dir):
	"""
	Descriptive statistics requested by Dr. Rifkin (no modeling):
	  1. % of tumors in each lobe, % right vs. left sided tumors
	  2. % early onset epilepsy (pre-operative seizure) vs. % late onset epilepsy (seizure after presentation)
	  3. most common ASM, % of patients treated with an ASM
	  4. average number of EEGs per patient
	  5. average (most common) EEG length category
	Writes retry_descriptives.txt and retry_descriptives.png to output_dir.
	"""
	output_dir = os.path.expanduser(output_dir)
	n_pts = len(patient_data)
	lines = []
	lines.append("RETRY descriptive statistics")
	lines.append("Number of patients included (N) = " + str(n_pts))
	lines.append("")

	# ---------- 1. Tumor location ----------
	lines.append("1. TUMOR LOCATION (% of all N patients; a patient can be counted in more than one lobe)")
	lobe_pct = {}
	for lobe in LOBES:
		n = int(checked(patient_data, 'loc_' + lobe).sum())
		lobe_pct[lobe] = pct(n, n_pts)
		lines.append("   " + lobe.ljust(14) + str(n).rjust(4) + "  (" + format(lobe_pct[lobe], ".1f") + "%)")
	any_lobe = pd.concat([checked(patient_data, 'loc_' + l) for l in LOBES], axis=1).any(axis=1)
	lines.append("   No lobe recorded: " + str(int((~any_lobe).sum())))
	lines.append("")

	left = pd.concat([checked(patient_data, 'loc_left_' + l) for l in LOBES], axis=1).any(axis=1)
	right = pd.concat([checked(patient_data, 'loc_right_' + l) for l in LOBES], axis=1).any(axis=1)
	side_counts = {
		'Left only': int((left & ~right).sum()),
		'Right only': int((right & ~left).sum()),
		'Both sides': int((left & right).sum()),
		'No side recorded': int((~left & ~right).sum()),
	}
	lines.append("   TUMOR SIDE (% of all N patients)")
	for k, v in side_counts.items():
		lines.append("   " + k.ljust(18) + str(v).rjust(4) + "  (" + format(pct(v, n_pts), ".1f") + "%)")
	lines.append("   Any left-sided (includes both sides):  " + str(int(left.sum())) + "  (" + format(pct(left.sum(), n_pts), ".1f") + "%)")
	lines.append("   Any right-sided (includes both sides): " + str(int(right.sum())) + "  (" + format(pct(right.sum(), n_pts), ".1f") + "%)")
	lines.append("")

	# ---------- 2. Early vs. late onset epilepsy ----------
	# early onset = seizure before surgery (preop_sz); late onset = seizure after initial presentation (late_sz)
	lines.append("2. EARLY vs. LATE ONSET EPILEPSY (% of all N patients; 'unknown' includes blank and Indeterminate)")
	onset_pct = {}
	for label, col in [('Early onset (pre-operative seizure)', 'preop_sz'), ('Late onset (seizure after presentation)', 'late_sz')]:
		n_yes = int((patient_data[col] == True).sum())
		n_no = int((patient_data[col] == False).sum())
		n_unk = n_pts - n_yes - n_no
		onset_pct[label] = pct(n_yes, n_pts)
		lines.append("   " + label)
		lines.append("      Yes: " + str(n_yes) + " (" + format(pct(n_yes, n_pts), ".1f") + "% of N; " + format(pct(n_yes, n_yes + n_no), ".1f") + "% of known)")
		lines.append("      No: " + str(n_no) + "   Unknown: " + str(n_unk))
	both = int(((patient_data['preop_sz'] == True) & (patient_data['late_sz'] == True)).sum())
	lines.append("   Both early and late seizures: " + str(both))
	lines.append("")

	# ---------- 3. Anti-seizure medications (ASM) ----------
	# ASMs are recorded on each EEG, so a patient counts as on a drug if ANY of their EEGs lists it
	lines.append("3. ANTI-SEIZURE MEDICATIONS (from the ASM field on each EEG record)")
	asm_cols = [c for c in eeg_data.columns if c.startswith('eeg_asm_')]
	drug_cols = [c for c in asm_cols if c not in ('eeg_asm_none', 'eeg_asm_other')]
	asm_flags = pd.DataFrame({c: checked(eeg_data, c) for c in asm_cols})
	asm_flags['study_id'] = eeg_data['study_id'].values
	by_patient = asm_flags.groupby('study_id').any()
	n_on_asm = int(by_patient[drug_cols + ['eeg_asm_other']].any(axis=1).sum())
	n_asm_recorded = int(by_patient.any(axis=1).sum())
	lines.append("   Patients on at least one ASM: " + str(n_on_asm) + " (" + format(pct(n_on_asm, n_pts), ".1f") + "% of all N; "
				 + format(pct(n_on_asm, n_asm_recorded), ".1f") + "% of the " + str(n_asm_recorded) + " patients with ASM data recorded)")
	lines.append("   Patients with ASM recorded as 'none' on every EEG: " + str(int((by_patient['eeg_asm_none'] & ~by_patient[drug_cols + ['eeg_asm_other']].any(axis=1)).sum())))
	drug_counts = by_patient[drug_cols].sum().sort_values(ascending=False)
	drug_counts.index = [c.replace('eeg_asm_', '') for c in drug_counts.index]
	lines.append("   Most common ASM: " + str(drug_counts.index[0]) + " (" + str(int(drug_counts.iloc[0])) + " patients, " + format(pct(drug_counts.iloc[0], n_pts), ".1f") + "% of all N)")
	lines.append("   Number of patients on each ASM (ever listed on an EEG):")
	for drug, n in drug_counts[drug_counts > 0].items():
		lines.append("      " + drug.ljust(18) + str(int(n)).rjust(4) + "  (" + format(pct(n, n_pts), ".1f") + "%)")
	lines.append("      " + "other (unspecified)".ljust(18) + str(int(by_patient['eeg_asm_other'].sum())).rjust(4))
	lines.append("")

	# ---------- 4. Number of EEGs per patient ----------
	lines.append("4. NUMBER OF EEGs PER PATIENT")
	eeg_per_pt = eeg_data.groupby('study_id').size().reindex(patient_data['study_id'], fill_value=0)
	lines.append("   Total EEGs: " + str(len(eeg_data)))
	lines.append("   Mean per patient (all N, including patients with 0 EEGs): " + format(eeg_per_pt.mean(), ".2f"))
	lines.append("   Median per patient: " + format(eeg_per_pt.median(), ".1f") + "   Range: " + str(int(eeg_per_pt.min())) + " to " + str(int(eeg_per_pt.max())))
	has_eeg = eeg_per_pt[eeg_per_pt > 0]
	lines.append("   Mean among the " + str(len(has_eeg)) + " patients with at least 1 EEG: " + format(has_eeg.mean(), ".2f"))
	lines.append("")

	# ---------- 5. EEG length ----------
	lines.append("5. EEG LENGTH (category, % of EEGs with a length recorded)")
	dur = eeg_data['eeg_duration'].dropna()
	dur_counts = dur.value_counts().reindex(DURATION_ORDER, fill_value=0)
	other_dur = dur[~dur.isin(DURATION_ORDER)]
	for cat, n in dur_counts.items():
		lines.append("   " + cat.ljust(22) + str(int(n)).rjust(4) + "  (" + format(pct(n, len(dur)), ".1f") + "%)")
	if len(other_dur) > 0:
		lines.append("   Unrecognized categories: " + str(len(other_dur)) + " (check data dictionary)")
	lines.append("   Missing length: " + str(int(eeg_data['eeg_duration'].isna().sum())))
	lines.append("   Most common length category: " + str(dur_counts.idxmax()) + " (" + format(pct(dur_counts.max(), len(dur)), ".1f") + "% of EEGs)")

	with open(os.path.join(output_dir, "retry_descriptives.txt"), "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")

	# ---------- Figure (horizontal bars with the value printed on each bar so no axis reading is needed) ----------
	def hbar (ax, labels, values, title, color, fmt="{:.1f}%"):
		y = list(range(len(labels)))[::-1]
		ax.barh(y, values, color=color, edgecolor='black', height=0.6)
		ax.set_yticks(y)
		ax.set_yticklabels(labels, fontsize=12)
		for yy, v in zip(y, values):
			ax.text(v + max(values) * 0.015, yy, fmt.format(v), va='center', fontsize=12)
		ax.set_xlim(0, max(values) * 1.18)
		ax.set_title(title, fontsize=13, fontweight='bold')
		ax.xaxis.set_visible(False)
		for s in ['top', 'right', 'bottom']:
			ax.spines[s].set_visible(False)
	fig, axes = plt.subplots(3, 2, figsize=(12, 15))
	hbar(axes[0, 0], LOBES, [lobe_pct[l] for l in LOBES], "Tumor location (% of patients)", 'steelblue')
	hbar(axes[0, 1], list(side_counts.keys()), [pct(v, n_pts) for v in side_counts.values()], "Tumor side (% of patients)", 'coral')
	hbar(axes[1, 0], ['Early onset (before surgery)', 'Late onset (after presentation)'], list(onset_pct.values()), "Epilepsy onset (% of all patients)", 'steelblue')
	top_drugs = drug_counts.head(8)
	hbar(axes[1, 1], list(top_drugs.index), [pct(v, n_pts) for v in top_drugs.values], "Most common ASMs (% of patients)", 'steelblue')
	counts = eeg_per_pt.value_counts().sort_index()
	ax = axes[2, 0]
	ax.bar(counts.index, counts.values, color='coral', edgecolor='black', width=0.8)
	for xx, v in zip(counts.index, counts.values):
		ax.text(xx, v + 2, str(int(v)), ha='center', fontsize=11)
	ax.set_xticks(list(range(0, int(eeg_per_pt.max()) + 1)))
	ax.set_title("EEGs per patient (number of patients)", fontsize=13, fontweight='bold')
	ax.set_xlabel("Number of EEGs a patient has", fontsize=12)
	ax.yaxis.set_visible(False)
	for s in ['top', 'right', 'left']:
		ax.spines[s].set_visible(False)
	hbar(axes[2, 1], list(dur_counts.index), [pct(v, len(dur)) for v in dur_counts.values], "EEG length category (% of EEGs)", 'steelblue')
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "retry_descriptives.png"), dpi=200)
	plt.close()

# Run on its own:  python3 retry_descriptives.py [data CSV] [data dictionary CSV]
# Results are saved next to the data CSV.
if __name__ == "__main__":
	import sys
	from retry_parser import redcap_parser
	data_file, dict_file = str(sys.argv[1]), str(sys.argv[2])
	patient_data, eeg_data = redcap_parser(data_file, dict_file)
	retry_descriptives(patient_data, eeg_data, os.path.dirname(os.path.abspath(data_file)))
