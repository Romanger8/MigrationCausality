# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 17:05:22 2020

@author: Administrator
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
#%matplotlib inline

# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests



    

# Preprocess data
path = 'C:/Users/Administrator/Documents/PG/02 Family/RG/'

def process_migr_data(core_path=path, migr_name='migr_with_lags.csv'):
    #migr_name = 'migr_clean_data.csv'
    migr_data = pd.read_csv(core_path + migr_name, sep=';')
    migr_data['d_l_stock_migr'] = np.log(migr_data['foreign_popul_stock']) - \
                                    np.log(migr_data['foreign_popul_stock_min_1'])
    migr_data['d2_l_stock_migr'] = migr_data['d_l_stock_migr'] - \
                                (np.log(migr_data['foreign_popul_stock_min_1']) - \
                                 np.log(migr_data['foreign_popul_stock_min_2']))
    migr_data.index = migr_data['Country'] + migr_data['Year'].astype(str)
    migr_cols = ['foreign_popul_stock', 'foreign_popul_stock_min_1', 
                 'foreign_popul_stock_min_2', 'foreign_popul_diff', 
                 'foreign_popul_pct_change', 'd_l_stock_migr', 'd2_l_stock_migr']
    migr_data = migr_data[migr_cols]
    return migr_data

def process_gdp_data(core_path=path, gdp_name='gdp_with_lag.csv'):
    #gdp_name = 'gdp_with_lag.csv'
    gdp_data = pd.read_csv(core_path + gdp_name, sep=';')
    gdp_data = gdp_data[gdp_data['Subject']=='GDP per capita, constant prices ']
    gdp_data.index = gdp_data['Country'] + gdp_data['Time'].astype(str)
    gdp_data_growth = gdp_data[gdp_data['Measure']=='Annual growth/change']
    gdp_data_growth['gdp_pct_growth'] = gdp_data_growth['Value']
    gdp_data_growth = gdp_data_growth[['Country', 'Time', 'gdp_pct_growth']]
    gdp_data_ind = gdp_data[gdp_data['Measure']=='Index']
    gdp_data_ind['gdp_index'] = gdp_data_ind['Value']
    gdp_data_ind['gdp_index_min1'] = gdp_data_ind['Value_min1']
    gdp_full = gdp_data_ind.merge(gdp_data_growth, how='outer', left_index=True, right_index=True)
    gdp_full['d_l_gdp'] = np.log(gdp_full['gdp_index']) - \
                        np.log(gdp_full['gdp_index_min1'])
    return gdp_full



def prepare_for_country(in_data, country):
    out_data = in_data.copy(deep=True)
    out_data = out_data[out_data['country']==country]
    out_data.index = out_data['TIME']
    
    return out_data

"""

# I. Test stationarity of each of the time series
# https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test
# https://machinelearningmastery.com/time-series-data-stationary-python/

# ser_gdp_pct_change
ser_gdp_pct_change = single_country_data['gdp_pct_growth'].dropna().values
adf_gdp_pct_change = adfuller(ser_gdp_pct_change)
print('RESULTS FOR GDP_PCT_CHANGE:')
print('ADF Statistic: %f' % adf_gdp_pct_change[0])
print('p-value: %f' % adf_gdp_pct_change[1])
print('Critical Values:')
for key, value in adf_gdp_pct_change[4].items():
	print('\t%s: %.3f' % (key, value))

# d_l_gdp
ser_d_l_gdp = single_country_data['d_l_gdp'].dropna().values
adf_d_l_gdp = adfuller(ser_d_l_gdp)
print('RESULTS FOR d_l_gdp:')
print('ADF Statistic: %f' % adf_d_l_gdp[0])
print('p-value: %f' % adf_d_l_gdp[1])
print('Critical Values:')
for key, value in adf_d_l_gdp[4].items():
	print('\t%s: %.3f' % (key, value))
    
# gdp_index
ser_gdp_index = single_country_data['gdp_index'].dropna().values
adf_gdp_index = adfuller(ser_gdp_index)
print('RESULTS FOR GDP_INDEX:')
print('ADF Statistic: %f' % adf_gdp_index[0])
print('p-value: %f' % adf_gdp_index[1])
print('Critical Values:')
for key, value in adf_gdp_index[4].items():
	print('\t%s: %.3f' % (key, value))

# foreign_popul_diff
ser_foreign_popul_diff = single_country_data['foreign_popul_diff'].dropna().values
adf_foreign_popul_diff = adfuller(ser_foreign_popul_diff)
print('RESULTS FOR FOREIGN_POPUL_DIFF:')
print('ADF Statistic: %f' % adf_foreign_popul_diff[0])
print('p-value: %f' % adf_foreign_popul_diff[1])
print('Critical Values:')
for key, value in adf_foreign_popul_diff[4].items():
	print('\t%s: %.3f' % (key, value))    

# foreign_popul_pct_change
ser_foreign_popul_pct_change = single_country_data['foreign_popul_pct_change'].dropna().values
adf_foreign_popul_pct_change = adfuller(ser_foreign_popul_pct_change)
print('RESULTS FOR FOREIGN_POPUL_PCT_CHANGE:')
print('ADF Statistic: %f' % adf_foreign_popul_pct_change[0])
print('p-value: %f' % adf_foreign_popul_pct_change[1])
print('Critical Values:')
for key, value in adf_foreign_popul_pct_change[4].items():
	print('\t%s: %.3f' % (key, value))    

# foreign_popul_pct_change 1st diff
ser_foreign_popul_pct_change_1diff = single_country_data['foreign_popul_pct_change'].diff().dropna().values
adf_foreign_popul_pct_change_1diff = adfuller(ser_foreign_popul_pct_change_1diff)
print('RESULTS FOR FOREIGN_POPUL_PCT_CHANGE 1ST DIFF:')
print('ADF Statistic: %f' % adf_foreign_popul_pct_change_1diff[0])
print('p-value: %f' % adf_foreign_popul_pct_change_1diff[1])
print('Critical Values:')
for key, value in adf_foreign_popul_pct_change_1diff[4].items():
            	print('\t%s: %.3f' % (key, value)) 
                
# d_l_stock_migr
ser_d_l_stock_migr = single_country_data['d_l_stock_migr'].dropna().values
adf_d_l_stock_migr = adfuller(ser_d_l_stock_migr)
print('RESULTS FOR d_l_stock_migr:')
print('ADF Statistic: %f' % adf_d_l_stock_migr[0])
print('p-value: %f' % adf_d_l_stock_migr[1])
print('Critical Values:')
for key, value in adf_d_l_stock_migr[4].items():
            	print('\t%s: %.3f' % (key, value))
    
# d2_l_stock_migr
ser_d2_l_stock_migr = single_country_data['d2_l_stock_migr'].dropna().values
adf_d2_l_stock_migr = adfuller(ser_d2_l_stock_migr)
print('RESULTS FOR d2_l_stock_migr:')
print('ADF Statistic: %f' % adf_d2_l_stock_migr[0])
print('p-value: %f' % adf_d2_l_stock_migr[1])
print('Critical Values:')
for key, value in adf_d2_l_stock_migr[4].items():
            	print('\t%s: %.3f' % (key, value))
                
# II. Try time series model with lags until residuals are normal and not autocorrelated

# II.a) GDP
tst = single_country_data['d_l_gdp'].dropna()

model = ARIMA(tst, order=(1,0,0))
model_fit = model.fit(disp=0)
# gdp_pct_change(t+1) = 1.7605 + 0.3434*gdp_pct_change(t)
# 2019: 2.44 +1 -0.75
# 2019: 1.7 3.44 
print('GDP AR MODEL RESULTS:') 
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())



model_res = ARIMA(residuals[0], order=(1,0,0))
model_fit_res = model_res.fit(disp=0)
# gdp_pct_change(t+1) = 1.7605 + 0.3434*gdp_pct_change(t)
# 2019: 2.44 +1 -0.75
# 2019: 1.7 3.44 
print('RESIDUAL SUMMARY: ', model_fit_res.summary())

# II.b) Migration
#single_country_data['foreign_popul_pct_change_1diff'] = single_country_data['foreign_popul_pct_change'].diff()
#tst_migr = single_country_data[['foreign_popul_pct_change_1diff', 'foreign_popul_pct_change', 'gdp_pct_growth']]
migr_ser = single_country_data['d2_l_stock_migr'].dropna()

migr_model = ARIMA(migr_ser, order=(1,0,0))
migr_model_fit = migr_model.fit(disp=0)
# gdp_pct_change(t+1) = 1.7605 + 0.3434*gdp_pct_change(t)
# 2019: 2.44 +1 -0.75
# 2019: 1.7 3.44
# MIGR AR MODEL RESULTS:
print('MIGR AR MODEL RESULTS:') 
print(migr_model_fit.summary())
# plot residual errors
migr_residuals = pd.DataFrame(migr_model_fit.resid)
migr_residuals.plot()
plt.show()
migr_residuals.plot(kind='kde')
plt.show()
print(migr_residuals.describe())



migr_model_res = ARIMA(migr_residuals[0], order=(1,0,0))
migr_model_fit_res = migr_model_res.fit(disp=0)
# gdp_pct_change(t+1) = 1.7605 + 0.3434*gdp_pct_change(t)
# 2019: 2.44 +1 -0.75
# 2019: 1.7 3.44 
print('RESIDUAL SUMMARY: ', migr_model_fit_res.summary())



# III. Test granger causality with the above defined lags
#gc_migr_data = single_country_data[['foreign_popul_pct_change_1diff', 'gdp_pct_growth']].dropna()
#gc_gdp_data = single_country_data[['gdp_pct_growth', 'foreign_popul_pct_change_1diff']].dropna()
gc_migr_data = single_country_data[['d_l_stock_migr', 'd_l_gdp']].dropna()
gc_gdp_data = single_country_data[['d_l_gdp', 'd_l_stock_migr']].dropna()

print("")
print('RUN GRANGER CAUSALITY IF GDP_GROWTH EXPLAINS FOREGN POPULATION CHANGE')
gc_res_migr = grangercausalitytests(gc_migr_data, [1])
print("")
print('RUN GRANGER CAUSALITY IF FOREGN POPULATION CHANGE EXPLAINS GDP_GROWTH')
gc_res_gdp = grangercausalitytests(gc_gdp_data, [1])
"""
# Load data
"""
f_name = 'merged_data.csv'
country_name = 'Netherlands'
min_year = 1989

migr_data_raw = pd.read_csv(core_path + f_name)

columns_clean = ['Country', 'Year', 'For_pop_inflow', 'gdp_growth']
migr_data_clean = migr_data_raw[columns_clean]
migr_data_clean = migr_data_clean[migr_data_clean['Country'] == country_name]
migr_data_clean = migr_data_clean[migr_data_clean['Year'] >= min_year]

stats, p = stats.normaltest(migr_data_clean['gdp_growth'].dropna())
print('Statistics=%.3f, p=%.3f' % (stats,p))
migr_data_clean['gdp_growth'].hist()
"""

class CausalityAnalysis:
    
    def __init__(self, country, max_lag=3, AR_order=1):
        self.country = country
        self.max_lag = max_lag
        self.AR_order = AR_order
        self.results = {}
        self.non_stationarity = {}
        
        
    def get_results(self, in_data):
        
        # Choose the correct country
        single_country_data = self.filter_country(in_data)
        self.tst = single_country_data
        # Test order of difference which will be stationary
        self.test_all_diferences(single_country_data)
        
        # Run Granger causality
        gdp_diff_level = self.results['stationarity_diff_levels']['gdp']
        migr_diff_level = self.results['stationarity_diff_levels']['migr']
        print('DIFF LEVELS: ', self.country, 'GDP: ', gdp_diff_level, 
              'MIGRATION', 'migr_diff_level')
        
        p_val_gdp, p_val_migr = self.test_granger_causality(single_country_data, gdp_diff_level, 
                                                   migr_diff_level)
                
        self.results['granger_causality_res'] = {}
        self.results['granger_causality_res']['gdp'] = p_val_gdp
        self.results['granger_causality_res']['migr'] = p_val_migr
        
        output = {}
        output['migr_diff'] = self.results['stationarity_diff_levels']['migr']
        output['gdp_diff'] = self.results['stationarity_diff_levels']['gdp']
        output['p_val_gdp_explained_by_migr'] = self.results['granger_causality_res']['gdp']
        output['p_val_migr_explained_by_gdp'] = self.results['granger_causality_res']['migr']
        
        self.results['final_output'] = output
        
    
    def filter_country(self, in_data):
        out_data = in_data.copy(deep=True)
        out_data = out_data[out_data['country']==self.country]
        out_data.index = out_data['TIME']
        return out_data
        
    
    def test_all_diferences(self, in_data):
        # Test order of difference which will be stationary
        # TODO: add 2-nd diff of GDP and 0 diff of migration and 0 diff of GDP
        # TODO: cross-check if/elif statements in this method
        col_names = ['d_l_gdp', 'd_l_stock_migr', 'd2_l_stock_migr']
        p_values = {}
        diff_levels = {}
        
        for col_name in col_names:
            p_values[col_name] = self.test_diff_order(in_data,
                                                           col_name)
            
        self.results['stationarity_p_values'] = p_values
        

        
        if p_values['d_l_gdp'] < 0.05: diff_levels['gdp'] = 1
        if p_values['d_l_gdp'] >= 0.05: diff_levels['gdp'] = 2
        
        if p_values['d_l_stock_migr'] < 0.05: diff_levels['migr'] = 1
        elif p_values['d2_l_stock_migr'] < 0.05: diff_levels['migr'] = 2
        else: diff_levels['migr'] = 2
        
        self.results['stationarity_diff_levels'] = diff_levels
        
        
        
        
    def test_diff_order(self, in_data, col_name):
        # d_l_gdp
        series_for_adf = in_data[col_name].dropna().values
        adf_results = adfuller(series_for_adf)
        #print('RESULTS FOR ' + col_name)
        #print('p-value: %f' % adf_results[1])
        
        return adf_results[1]
    
    def test_AR():
        pass
    
    def test_granger_causality(self, in_data, gdp_d, migr_d):
        if gdp_d == 1: gdp_name = 'd_l_gdp'
        if gdp_d == 2: gdp_name = 'd2_l_gdp'
        if migr_d == 1: migr_name = 'd_l_stock_migr'
        if migr_d == 2: migr_name = 'd2_l_stock_migr'
        
        gc_migr_data = in_data[[migr_name, gdp_name]].dropna()
        gc_gdp_data = in_data[[gdp_name, migr_name]].dropna()
        
        p_val_gdp = 1
        p_val_migr = 1
        
        for i in range(2):
            print("")
            print('RUN GRANGER CAUSALITY IF GDP_GROWTH EXPLAINS FOREGN POPULATION CHANGE')
            gc_res_migr = grangercausalitytests(gc_migr_data, [i+1])
            
            print(gc_res_migr)
            
            p1 = gc_res_migr[i+1][0]['ssr_ftest'][1]
            p2 = gc_res_migr[i+1][0]['ssr_chi2test'][1]
            p3 = gc_res_migr[i+1][0]['lrtest'][1]
            p4 = gc_res_migr[i+1][0]['params_ftest'][1]
            p = min(p1, p2, p3, p4)
            p_val_migr = min(p_val_migr, p)
            
            print("")
            print('RUN GRANGER CAUSALITY IF FOREGN POPULATION CHANGE EXPLAINS GDP_GROWTH')
            gc_res_gdp = grangercausalitytests(gc_gdp_data, [i+1])
            
            p1 = gc_res_gdp[i+1][0]['ssr_ftest'][1]
            p2 = gc_res_gdp[i+1][0]['ssr_chi2test'][1]
            p3 = gc_res_gdp[i+1][0]['lrtest'][1]
            p4 = gc_res_gdp[i+1][0]['params_ftest'][1]
            p = min(p1, p2, p3, p4)
            p_val_gdp = min(p_val_gdp, p)
            
            
            #p_val_gdp = 2
            #p_val_migr = 3
                       
        return p_val_gdp, p_val_migr
    
    
if __name__ == '__main__':
    # Load and process migration data
    migr_df = process_migr_data()
    
    # Load and process gdp data
    gdp_df = process_gdp_data()
    
    # Join gdp and migration data and keep only useful columns
    merged_data = migr_df.merge(gdp_df, how='outer', left_index=True, right_index=True)
    merged_data = merged_data.rename(columns={"Country_x": "country"})
    useful_columns = ['foreign_popul_stock', 'foreign_popul_stock_min_1',
       'foreign_popul_stock_min_2', 'd_l_stock_migr', 'd2_l_stock_migr',
       'country_year', 'country', 'Subject', 'Measure', 'TIME', 'gdp_index', 
       'gdp_index_min1', 'd_l_gdp']
    merged_data = merged_data[useful_columns]
    merged_data = merged_data[merged_data.TIME >= 1980]
    merged_data = merged_data[merged_data.TIME <= 2005]
    
    country_list = ['Germany', 'United Kingdom', 'Norway']
    
    granger_causality_results = {}
    for c in country_list:
        causality_results = CausalityAnalysis(c)
        causality_results.get_results(merged_data)
        granger_causality_results[c] = causality_results.results['final_output']
        # TODO: Check for 4 countries based on deltas and logs as in the paper 'SSRN-id2258385 to compare granger caus results'
        # TODO: debug for Iceland
