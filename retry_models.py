import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import numpy as np
from tabulate import tabulate

def logistic_regression (patients, outcome, predictor_cols, model_name, output_dir):

	model_df = patients[predictor_cols + [outcome]].copy()

	for c in predictor_cols:
		if str(model_df[c].dtype) == "boolean":
			model_df[c] = model_df[c].astype("Float64")
		model_df[c] = pd.to_numeric(model_df[c], errors="coerce")

	if str(model_df[outcome].dtype) == "boolean":
		model_df[outcome] = model_df[outcome].astype("Float64")
	model_df[outcome] = pd.to_numeric(model_df[outcome], errors="coerce")

	model_df = model_df.dropna()

	x = sm.add_constant(model_df[predictor_cols].astype(float), has_constant="add")
	y = model_df[outcome].astype(int)

	model = sm.Logit(y,x).fit(disp=False)
	
	params = model.params
	conf = model.conf_int()
	odds_ratios = np.exp(params)
	odds_ci = np.exp(conf)

	or_table = pd.DataFrame({
		"OR": odds_ratios,
		"2.5%": odds_ci[0],
		"97.5%": odds_ci[1]
	})

	with open(os.path.join(output_dir, f"{model_name}.txt"), "w") as f:
		f.write(model.summary().as_text())
		f.write("\n\n")
		f.write(tabulate(or_table, headers='keys', tablefmt='github', floatfmt=".3f"))

	results = {}
	for var in predictor_cols:
		if var in odds_ratios.index:
			results[var] = (
				odds_ratios[var],
				odds_ci.loc[var, 0],
				odds_ci.loc[var, 1]
			)
	return results


def ordinal_regression (patients, outcome, predictor_cols, model_name, output_dir):

	model_df = patients[predictor_cols + [outcome]].copy()

	for c in predictor_cols:
		if str(model_df[c].dtype) == "boolean":
			model_df[c] = model_df[c].astype("Float64")
		model_df[c] = pd.to_numeric(model_df[c], errors="coerce")

	if str(model_df[outcome].dtype) == "boolean":
		model_df[outcome] = model_df[outcome].astype("Float64")
	model_df[outcome] = pd.to_numeric(model_df[outcome], errors="coerce")

	model_df = model_df.dropna()

	x = model_df[predictor_cols].astype(float)
	y_int = model_df[outcome].astype(int)
	y = pd.Categorical(y_int, categories=sorted(y_int.unique()), ordered=True)

	ord_model = OrderedModel(y, x, distr='logit')
	ord_res = ord_model.fit(method='bfgs', disp=False)

	# Odds ratios for predictors only (exclude threshold/cutpoint params like "1/2")
	params = ord_res.params
	conf = ord_res.conf_int()

	predictor_mask = ~params.index.str.contains('/')
	predictor_params = params[predictor_mask]
	predictor_conf = conf.loc[predictor_params.index]

	odds_ratios = np.exp(predictor_params)
	odds_ci = np.exp(predictor_conf)

	or_table = pd.DataFrame({
		"OR": odds_ratios,
		"2.5%": odds_ci[0],
		"97.5%": odds_ci[1]})
	
	with open(os.path.join(output_dir, f"{model_name}.txt"), "w") as f:
		f.write(ord_res.summary().as_text())
		f.write("\n\n")
		f.write(tabulate(or_table, headers='keys', tablefmt='github', floatfmt=".3f"))

	results = {}
	for var in predictor_cols:
		if var in odds_ratios.index:
			results[var] = (
				odds_ratios[var],
				odds_ci.loc[var, 0],
				odds_ci.loc[var, 1]
			)
	return results