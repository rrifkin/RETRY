import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import numpy as np

def logistic_regression (patients, outcome, predictor_cols):

	# Logistic regression to identify predictors of late-onset seizures in patients without seizures at baseline
	x = patients[predictor_cols].copy()
	y = patients.loc[:, outcome].map({'Yes': 1, 'No': 0})

	# Drop rows where either x or y has NaN
	valid_idx = x.dropna().index.intersection(y.dropna().index)
	x = x.loc[valid_idx]
	y = y.loc[valid_idx]	
	x = sm.add_constant(x)

	model = sm.Logit(y,x).fit()
	print(model.summary())

	params = model.params
	conf = model.conf_int()
	odds_ratios = np.exp(params)
	odds_ci = np.exp(conf)

	print(pd.DataFrame({
		"OR": odds_ratios,
		"2.5%": odds_ci[0],
		"97.5%": odds_ci[1]
	}))

def ordinal_regression (patients, outcome, predictor_cols):

	model_df = patients[predictor_cols + [outcome]].dropna().copy()

	x = model_df[predictor_cols]
	y_int = model_df[outcome].astype(int)
	y = pd.Categorical(y_int, categories=sorted(y_int.unique()), ordered=True)

	ord_model = OrderedModel(y, x, distr='logit')
	ord_res = ord_model.fit(method='bfgs', disp=False)
	print(ord_res.summary())

	# Odds ratios for predictors only (exclude threshold/cutpoint params like "1/2")
	params = ord_res.params
	conf = ord_res.conf_int()

	predictor_mask = ~params.index.str.contains('/')
	predictor_params = params[predictor_mask]
	predictor_conf = conf.loc[predictor_params.index]

	odds_ratios = np.exp(predictor_params)
	odds_ci = np.exp(predictor_conf)

	print(pd.DataFrame({
		"OR": odds_ratios,
		"2.5%": odds_ci[0],
		"97.5%": odds_ci[1]}))