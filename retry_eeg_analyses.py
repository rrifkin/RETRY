import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from tabulate import tabulate

# Dr. Rifkin's two modeling questions:
#   1. Is EEG hyperexcitability at ANY time point associated with clinical seizures at a LATER time point?
#   2. Is there an association between tumor location and the location of EEG hyperexcitability?
# Both functions write a .txt (numbers) and a .png (figure) to output_dir.

# EEG yes/no fields that count as hyperexcitability (same definition as retry_parser.find_hyperexcitability)
HYPER_FIELDS = {'gpd': 'gpd_yn', 'lpd': 'lpd_yn', 'bipd': 'bipd_yn', 'lrda': 'lrda_yn', 'esz': 'sz_yn', 'ied': 'epi_disch_yn'}

LOBES = ['Frontal', 'Temporal', 'Parietal', 'Occipital']	# lobes that exist for BOTH the tumor and the EEG forms
LOCATED_FINDINGS = {'lpd': 'LPD', 'lrda': 'LRDA', 'sz': 'seizure on EEG', 'epi_disch': 'epileptiform discharge'}	# EEG findings with a left/right + lobe location

def checked (df, col):
	return pd.to_numeric(df[col], errors='coerce').fillna(0).eq(1)

def pct (n, total):
	return (n / total * 100) if total > 0 else float('nan')

def fmt_p (p):
	return "<0.001" if p < 0.001 else format(p, ".3f")

# ----------------------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------------------

# one row per EEG, with a real date and TRUE/FALSE flags for each hyperexcitability pattern
def prepare_eeg (eeg_data):
	e = eeg_data.copy()
	e['eeg_date'] = pd.to_datetime(e['eeg_report_date'].replace('', np.nan), errors='coerce')
	for name, col in HYPER_FIELDS.items():
		e[name] = e[col].astype(str).str.strip().eq('1')
	e['pd'] = e[['gpd', 'lpd', 'bipd']].any(axis=1)
	e['any_disc'] = e[['gpd', 'lpd', 'bipd', 'ied']].any(axis=1)
	e['hyperexcitability'] = e[['gpd', 'lpd', 'bipd', 'lrda', 'esz', 'ied']].any(axis=1)
	return e

# logistic regression that also returns p-values and counts (retry_models.logistic_regression only returns ORs)
def fit_logit (df, outcome, predictors):
	d = df[predictors + [outcome]].copy()
	for col in d.columns:
		if str(d[col].dtype) == "boolean":
			d[col] = d[col].astype("Float64")
		d[col] = pd.to_numeric(d[col], errors="coerce")
	d = d.dropna()
	x = sm.add_constant(d[predictors].astype(float), has_constant="add")
	y = d[outcome].astype(int)
	try:
		m = sm.Logit(y, x).fit(disp=False)
	except Exception as err:
		return None, len(d), int(y.sum()), str(err)
	conf = np.exp(m.conf_int())
	table = pd.DataFrame({"OR": np.exp(m.params), "lo": conf[0], "hi": conf[1], "p": m.pvalues})
	return table, len(d), int(y.sum()), ""

def model_text (title, table, n, events, err):
	out = [title, "   patients in model: " + str(n) + "   with late seizure: " + str(events)]
	if table is None:
		out.append("   model did not fit: " + err)
		return out
	shown = table.drop('const', errors='ignore')
	rows = [[name, format(r['OR'], ".2f"), format(r['lo'], ".2f") + " to " + format(r['hi'], ".2f"), fmt_p(r['p'])] for name, r in shown.iterrows()]
	out.append(tabulate(rows, headers=["variable", "odds ratio", "95% CI", "p"], tablefmt="github"))
	return out

# forest plot on a LOG axis so very wide intervals are not cut off
def forest (ax, table, title):
	t = table.drop('const', errors='ignore')
	ypos = np.arange(len(t))[::-1]
	ax.errorbar(t['OR'], ypos, xerr=[t['OR'] - t['lo'], t['hi'] - t['OR']], fmt='o', markersize=7, capsize=4, color='steelblue')
	ax.axvline(1, color='red', linestyle='--', linewidth=1.3, alpha=0.7)
	ax.set_xscale('log')
	ax.set_yticks(ypos)
	ax.set_yticklabels(t.index)
	ax.set_xlabel("Odds ratio (95% CI), log scale")
	ax.set_title(title, fontsize=10)
	ax.grid(axis='x', alpha=0.3)

# one-panel forest plot for the timing models: plain tick labels, odds ratio and p-value printed beside each row
PLOT_LABELS = {'hyperexcitability': 'Any hyperexcitability', 'any_disc': 'Any discharge (GPD/LPD/BiPD/IED)', 'lrda': 'LRDA',
			   'pd': 'Periodic discharges (PD)', 'ied': 'IED', 'ever_hyper': 'Hyperexcitability on any EEG'}
def timing_forest (results, n_patients, path):
	from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
	fig, ax = plt.subplots(figsize=(10, 6.4))
	y, last, ticks, names = 0, None, [], []
	for title, table in results:
		if table is None:
			continue
		group = title.split(".")[0] if title[:2] in ("A.", "B.", "C.") else "Reference"
		group = {"A": "Model A (one combined measure)", "B": "Model B", "C": "Model C", "Reference": "Reference: EEG dates ignored"}[group]
		y -= 0.8 if last else 0
		ax.text(-0.02, y, group, fontsize=11, fontweight='bold', va='center', ha='right', transform=ax.get_yaxis_transform())
		y -= 1
		last = group
		for var, r in table.drop(['const', 'age'], errors='ignore').iterrows():
			color = '#777777' if group.startswith("Reference") else '#1f5fa8'
			ax.errorbar(r['OR'], y, xerr=[[r['OR'] - r['lo']], [r['hi'] - r['OR']]], fmt='o', ms=8, capsize=5, color=color, lw=2)
			ax.text(14, y, format(r['OR'], ".2f") + " (" + format(r['lo'], ".2f") + " to " + format(r['hi'], ".2f") + ")  p = " + format(r['p'], ".2f"), va='center', fontsize=11)
			ticks.append(y)
			names.append(PLOT_LABELS.get(var, var))
			y -= 1
	ax.axvline(1, color='red', ls='--', lw=1.5)
	ax.set_xscale('log')
	ax.set_xlim(0.05, 12)
	ax.xaxis.set_major_locator(FixedLocator([0.1, 0.25, 0.5, 1, 2, 4, 8]))
	ax.xaxis.set_major_formatter(FixedFormatter(['0.1', '0.25', '0.5', '1', '2', '4', '8']))
	ax.xaxis.set_minor_locator(NullLocator())
	ax.set_yticks(ticks)
	ax.set_yticklabels(names, fontsize=11)
	ax.set_ylim(y + 0.2, 0.7)
	ax.tick_params(axis='x', labelsize=12)
	ax.grid(axis='x', alpha=0.3)
	ax.set_xlabel("Odds ratio of a later seizure (log scale: equal distances mean equal multiplication)\n<--- less likely          1 = no difference          more likely --->", fontsize=11)
	ax.set_title("EEG hyperexcitability BEFORE the first seizure vs later seizure (" + str(n_patients) + " patients, adjusted for age)", fontsize=12)
	ax.text(14, 0.35, "Odds ratio (95% CI)  p", fontsize=11, fontweight='bold', va='center')
	ax.spines[['top', 'right']].set_visible(False)
	plt.subplots_adjust(left=0.34, right=0.7, bottom=0.2, top=0.9)
	plt.savefig(path, dpi=200)
	plt.close()

# ----------------------------------------------------------------------------------------------
# Question 1: does hyperexcitability come BEFORE a later clinical seizure?
# ----------------------------------------------------------------------------------------------
def hyperexcitability_before_seizure (patient_data, eeg_data, output_dir):
	"""
	Who is analyzed: patients with NO seizure before surgery (preop_sz == No), because they are the ones in whom a
	biomarker could predict a FIRST seizure. Outcome: late seizure (yes/no).
	Timing rule: for a patient who had a late seizure, only EEGs dated BEFORE the first late seizure count.
	             For a patient who never had one, every EEG counts.
	"""
	output_dir = os.path.expanduser(output_dir)
	e = prepare_eeg(eeg_data)
	p = patient_data.copy()
	p['sz_date'] = pd.to_datetime(p['late_sz_date'].replace('', np.nan), errors='coerce')
	lines = ["QUESTION 1: Is EEG hyperexcitability at any time point associated with a LATER clinical seizure?", ""]

	# ---- who gets in (every step is counted so the numbers can be explained) ----
	c = p[p['preop_sz'] == False].copy()
	lines.append("HOW THE GROUP WAS BUILT")
	lines.append("   Patients with no pre-operative seizure: " + str(len(c)))
	n_unknown = int(c['late_sz'].isna().sum())
	c = c[c['late_sz'].notna()].copy()
	c['late'] = c['late_sz'].astype(bool)
	lines.append("   minus unknown late-seizure status: " + str(n_unknown) + "  -> " + str(len(c)) + " (" + str(int(c['late'].sum())) + " with a late seizure, " + str(int((~c['late']).sum())) + " without)")
	no_date = c['late'] & c['sz_date'].isna()
	lines.append("   minus late-seizure patients with no seizure date: " + str(int(no_date.sum())))
	c = c[~no_date].copy()

	# EEGs that count: all EEGs for seizure-free patients, only EEGs before the first seizure for the rest
	m = e.merge(c[['study_id', 'late', 'sz_date']], on='study_id', how='inner')
	m = m[(~m['late']) | (m['eeg_date'] < m['sz_date'])]
	flags = ['pd', 'ied', 'lrda', 'esz', 'any_disc', 'hyperexcitability']
	agg = m.groupby('study_id').agg(n_eeg=('eeg_date', 'size'), **{k: (k, 'any') for k in flags}).reset_index()
	c = c.merge(agg, on='study_id', how='left')
	n_no_eeg = int(c['n_eeg'].isna().sum())
	ca = c[c['n_eeg'].notna()].copy()
	n_no_eeg_late = int((c['n_eeg'].isna() & c['late']).sum())
	lines.append("   minus patients with no usable EEG (none at all, or none before their seizure): " + str(n_no_eeg)
				 + "  (" + str(n_no_eeg_late) + " of them had a late seizure, so their EEGs all came on or after the seizure)")
	lines.append("   FINAL GROUP: " + str(len(ca)) + " patients (" + str(int(ca['late'].sum())) + " later had a seizure, " + str(int((~ca['late']).sum())) + " did not)")
	lines.append("   Makeup of the final group: WHO grade 4 = " + str(int((ca['who'] == 4).sum())) + " of " + str(int(ca['who'].notna().sum())) + " with a known grade; IDH-mutant = "
				 + str(int((ca['idh'] == True).sum())) + " of " + str(int(ca['idh'].notna().sum())) + " with known IDH status")
	lines.append("")

	# ---- simple 2x2: hyperexcitability before the seizure vs. late seizure ----
	hy = ca['hyperexcitability'].astype(bool)
	a, b = int((hy & ca['late']).sum()), int((hy & ~ca['late']).sum())
	cc, d = int((~hy & ca['late']).sum()), int((~hy & ~ca['late']).sum())
	odds, p_f = fisher_exact([[a, b], [cc, d]])
	lines.append("SIMPLE COMPARISON (no adjustment)")
	lines.append("   Hyperexcitability on a qualifying EEG:     " + str(a + b) + " patients -> late seizure in " + str(a) + " (" + format(pct(a, a + b), ".1f") + "%)")
	lines.append("   No hyperexcitability on qualifying EEGs:   " + str(cc + d) + " patients -> late seizure in " + str(cc) + " (" + format(pct(cc, cc + d), ".1f") + "%)")
	lines.append("   Fisher exact test p = " + fmt_p(p_f))
	lines.append("")

	# ---- adjusted models ----
	# Dr. Rifkin's other models adjust for IDH, WHO grade and age. In THIS group that is not possible: nearly everyone is grade 4
	# and the few IDH-mutant patients never had a late seizure, so the model cannot estimate those two (odds ratios of 0 or millions).
	# Age is the only covariate with enough variation, so it is the only one used here.
	covs = ['age']
	specs = [
		("A. Any hyperexcitability BEFORE the seizure", ['hyperexcitability'] + covs),
		("B. Specific patterns (any_disc, lrda) BEFORE the seizure", ['any_disc', 'lrda'] + covs),
		("C. Specific patterns (pd, ied, lrda) BEFORE the seizure", ['pd', 'ied', 'lrda'] + covs),
	]
	results = []
	lines.append("MODELS (logistic regression, adjusted for age only; odds ratio above 1 = higher odds of a later seizure)")
	for title, preds in specs:
		table, n, events, err = fit_logit(ca, 'late', preds)
		results.append((title, table))
		lines += model_text("   " + title, table, n, events, err)
		lines.append("")

	# reference model: same patients, but hyperexcitability on ANY EEG, ignoring the dates (how the older 'ever' models work)
	ever = e.groupby('study_id')['hyperexcitability'].any()
	ca['ever_hyper'] = ca['study_id'].map(ever)
	table, n, events, err = fit_logit(ca, 'late', ['ever_hyper'] + covs)
	results.append(("Reference: hyperexcitability on ANY EEG (timing ignored)", table))
	lines += model_text("   Reference: same patients, hyperexcitability on ANY EEG, timing ignored", table, n, events, err)
	lines.append("")

	# ---- descriptive: for patients who did have a late seizure, when did hyperexcitability first appear? ----
	lt = p[(p['preop_sz'] == False) & (p['late_sz'] == True) & p['sz_date'].notna()].copy()
	first_h = e[e['hyperexcitability']].groupby('study_id')['eeg_date'].min()
	has_eeg = lt['study_id'].isin(e['study_id'])
	lt['first_h'] = lt['study_id'].map(first_h)
	before = lt['first_h'] < lt['sz_date']
	after = lt['first_h'] >= lt['sz_date']
	never = has_eeg & lt['first_h'].isna()
	lines.append("TIMING IN PATIENTS WHO HAD A LATE SEIZURE (n = " + str(len(lt)) + " with a seizure date, no pre-op seizure)")
	lines.append("   Hyperexcitability first seen BEFORE the first late seizure: " + str(int(before.sum())))
	lines.append("   Hyperexcitability first seen on or AFTER the seizure date:  " + str(int(after.sum())))
	lines.append("   No hyperexcitability on any EEG:                            " + str(int(never.sum())))
	lines.append("   No EEG at all:                                              " + str(int((~has_eeg).sum())))
	lead = (lt.loc[before, 'sz_date'] - lt.loc[before, 'first_h']).dt.days
	if len(lead) > 0:
		lines.append("   Among those with hyperexcitability first: it appeared a median of " + format(lead.median(), ".0f") + " days before the seizure (middle half: "
					 + format(lead.quantile(0.25), ".0f") + " to " + format(lead.quantile(0.75), ".0f") + " days)")
	lines.append("")
	lines.append("NOTES")
	lines.append("   - An EEG dated the same day as the seizure is NOT counted as 'before' (it may have been ordered because of the seizure).")
	lines.append("   - Seizure-free patients can have EEGs at any time, but seizure patients only count EEGs before the seizure, so the two groups are not observed")
	lines.append("     for exactly the same length of time. A time-to-event analysis would handle this more rigorously.")

	with open(os.path.join(output_dir, "retry_hyperexcitability_timing.txt"), "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")

	timing_forest(results, len(ca), os.path.join(output_dir, "retry_hyperexcitability_timing.png"))

# ----------------------------------------------------------------------------------------------
# Question 2: does tumor location match the location of EEG hyperexcitability?
# ----------------------------------------------------------------------------------------------
def tumor_vs_eeg_location (patient_data, eeg_data, output_dir):
	"""
	Uses only EEG findings that have a left/right + lobe location recorded (LPD, LRDA, seizure, epileptiform discharge).
	GPD and BiPD are generalized/bilateral, so they have no location and are left out.
	Tumor lobes: frontal, temporal, parietal, occipital (insula, subcortical, infratentorial have no matching EEG lobe).
	"""
	output_dir = os.path.expanduser(output_dir)
	e = eeg_data.copy()
	p = patient_data.copy()
	lines = ["QUESTION 2: Is tumor location associated with the location of EEG hyperexcitability?", ""]

	def eeg_cols (t, side, lobe=''):
		return [c for c in e.columns if c.startswith(t + '_' + side + '_' + lobe)]

	# per EEG: is there a finding of any located type on this side (any lobe), or in this lobe+side?
	def any_cols (cols):
		return pd.concat([checked(e, c) for c in cols], axis=1).any(axis=1)
	e['_sid'] = e['study_id']
	side_flags, lobe_flags, type_side = {}, {}, {}
	for side in ['left', 'right']:
		for t in LOCATED_FINDINGS:
			type_side[(t, side)] = any_cols(eeg_cols(t, side))
		side_flags[side] = pd.concat([type_side[(t, side)] for t in LOCATED_FINDINGS], axis=1).any(axis=1)
		for lobe in LOBES:
			cols = [c for t in LOCATED_FINDINGS for c in eeg_cols(t, side, lobe)]
			lobe_flags[(side, lobe)] = any_cols(cols)

	# per patient: did ANY of their EEGs show it?
	def per_patient (series):
		return series.groupby(e['_sid']).any()
	pt = pd.DataFrame({'study_id': p['study_id'].values})
	has_eeg = pt['study_id'].isin(e['study_id'])
	def pmap (series):
		return pt['study_id'].map(per_patient(series)).fillna(False).astype(bool)
	for side in ['left', 'right']:
		pt['eeg_' + side] = pmap(side_flags[side])
		for t in LOCATED_FINDINGS:
			pt['eeg_' + t + '_' + side] = pmap(type_side[(t, side)])
		for lobe in LOBES:
			pt['eeg_' + side + '_' + lobe] = pmap(lobe_flags[(side, lobe)])
	all_lobes = ['Frontal', 'Temporal', 'Insula', 'Parietal', 'Occipital', 'Subcortical', 'Infratentorial']
	for side in ['left', 'right']:
		pt['tumor_' + side] = pd.concat([checked(p, 'loc_' + side + '_' + l) for l in all_lobes], axis=1).any(axis=1).values
		for lobe in LOBES:
			pt['tumor_' + side + '_' + lobe] = checked(p, 'loc_' + side + '_' + lobe).values
	pt = pt[has_eeg.values].copy()
	lines.append("Patients with at least one EEG: " + str(len(pt)))
	lines.append("")

	# ---- A. SIDE: is a one-sided EEG finding on the same side as a one-sided tumor? ----
	lines.append("A. SIDE MATCH (patients with a tumor on one side only AND EEG findings on one side only)")
	def side_table (left_col, right_col):
		tl = pt['tumor_left'] & ~pt['tumor_right']
		tr = pt['tumor_right'] & ~pt['tumor_left']
		el = pt[left_col] & ~pt[right_col]
		er = pt[right_col] & ~pt[left_col]
		ll, lr = int((tl & el).sum()), int((tl & er).sum())
		rl, rr = int((tr & el).sum()), int((tr & er).sum())
		return ll, lr, rl, rr
	rows = []
	for label, lc, rc in [("Any located finding", 'eeg_left', 'eeg_right')] + [(name, 'eeg_' + t + '_left', 'eeg_' + t + '_right') for t, name in LOCATED_FINDINGS.items()]:
		ll, lr, rl, rr = side_table(lc, rc)
		n = ll + lr + rl + rr
		if n > 0 and min(ll + lr, rl + rr, ll + rl, lr + rr) > 0:
			odds, pv = fisher_exact([[ll, lr], [rl, rr]])
			rows.append([label, n, ll + rr, format(pct(ll + rr, n), ".1f") + "%", format(odds, ".2f"), fmt_p(pv)])
		else:
			rows.append([label, n, ll + rr, format(pct(ll + rr, n), ".1f") + "%" if n else "n/a", "n/a", "n/a"])
	lines.append(tabulate(rows, headers=["EEG finding", "patients", "same side", "% same side", "odds ratio", "Fisher p"], tablefmt="github"))
	ll, lr, rl, rr = side_table('eeg_left', 'eeg_right')
	lines.append("   Any located finding, counts: tumor LEFT with EEG left " + str(ll) + ", EEG right " + str(lr) + "; tumor RIGHT with EEG left " + str(rl) + ", EEG right " + str(rr))
	lines.append("   (50% same side would be what chance alone gives if side did not matter and both sides were equally common)")
	lines.append("")

	# ---- B. LOBE + SIDE: tumor in this lobe/side vs. an EEG finding in the same lobe/side ----
	lines.append("B. LOBE-BY-LOBE MATCH (all patients with an EEG; each row compares tumor in that lobe+side with an EEG finding in that same lobe+side)")
	cells, rows, pvals, plot_rows = [], [], [], []
	for side in ['left', 'right']:
		for lobe in LOBES:
			t, f = pt['tumor_' + side + '_' + lobe], pt['eeg_' + side + '_' + lobe]
			a, b = int((t & f).sum()), int((t & ~f).sum())
			cc, d = int((~t & f).sum()), int((~t & ~f).sum())
			odds, pv = fisher_exact([[a, b], [cc, d]])
			cells.append((side.capitalize() + " " + lobe, a, b, cc, d, odds, pv))
			pvals.append(pv)
	adj = multipletests(pvals, method='holm')[1]
	for (name, a, b, cc, d, odds, pv), pa in zip(cells, adj):
		rows.append([name, a + b, format(pct(a, a + b), ".1f") + "%", cc + d, format(pct(cc, cc + d), ".1f") + "%", format(odds, ".2f"), fmt_p(pv), fmt_p(pa)])
		plot_rows.append((name, pct(a, a + b), pct(cc, cc + d)))
	lines.append(tabulate(rows, headers=["lobe + side", "with tumor there", "% with EEG finding there", "no tumor there", "% with EEG finding there", "odds ratio", "Fisher p", "p (Holm adjusted)"], tablefmt="github"))
	lines.append("")
	lines.append("NOTES")
	lines.append("   - 'EEG finding' = any LPD, LRDA, seizure or epileptiform discharge with that lobe and side ticked, on any of the patient's EEGs.")
	lines.append("   - Eight lobe/side comparisons are run, so the Holm-adjusted p-value is the one to trust; the raw p-value is more likely to be a chance finding.")
	lines.append("   - Patients with more EEGs have more chances to show a finding, and findings marked 'unknown / poorly localized' are not counted in the lobe comparison.")
	lines.append("   - No adjustment for grade, IDH or age here; this is a description of how well the two locations line up.")

	with open(os.path.join(output_dir, "retry_location_match.txt"), "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")

	fig, ax = plt.subplots(figsize=(10, 6))
	x = np.arange(len(plot_rows))
	ax.bar(x - 0.2, [r[1] for r in plot_rows], width=0.4, color='coral', edgecolor='black', label="Tumor in this lobe/side")
	ax.bar(x + 0.2, [r[2] for r in plot_rows], width=0.4, color='steelblue', edgecolor='black', label="No tumor in this lobe/side")
	ax.set_xticks(x)
	ax.set_xticklabels([r[0].replace(' ', chr(10)) for r in plot_rows], fontsize=12)
	for i, r in enumerate(plot_rows):
		ax.text(i - 0.2, r[1] + 0.8, format(r[1], ".0f") + "%", ha='center', fontsize=11)
		ax.text(i + 0.2, r[2] + 0.8, format(r[2], ".0f") + "%", ha='center', fontsize=11)
	ax.set_ylabel("% of patients with an EEG finding\nin that lobe and side", fontsize=12)
	ax.set_title("EEG hyperexcitability location vs. tumor location")
	ax.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "retry_location_match.png"), dpi=300)
	plt.close()

# Run on its own:  python3 retry_eeg_analyses.py [data CSV] [data dictionary CSV]
# Results are saved next to the data CSV.
if __name__ == "__main__":
	import sys
	from retry_parser import redcap_parser
	data_file, dict_file = str(sys.argv[1]), str(sys.argv[2])
	patient_data, eeg_data = redcap_parser(data_file, dict_file)
	out = os.path.dirname(os.path.abspath(data_file))
	hyperexcitability_before_seizure(patient_data, eeg_data, out)
	tumor_vs_eeg_location(patient_data, eeg_data, out)
