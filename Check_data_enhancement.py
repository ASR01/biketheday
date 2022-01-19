import numpy    as np
import pandas   as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns

from sklearn import pipeline      # Pipeline
from sklearn import preprocessing # OrdinalEncoder, LabelEncoder
from sklearn import impute
from sklearn import compose
from sklearn import model_selection # train_test_split
from sklearn import metrics         # accuracy_score, balanced_accuracy_score, plot_confusion_matrix

from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import RandomForestRegressor
from sklearn.ensemble      import ExtraTreesRegressor
from sklearn.ensemble      import AdaBoostRegressor
from sklearn.ensemble      import GradientBoostingRegressor
from xgboost               import XGBRegressor
from lightgbm              import LGBMRegressor
from catboost              import CatBoostRegressor

#Variables 

datafile = './data/london_merged.csv'

def generate_data(data):
    
    gen_data = data
    
    for season in data['season'].unique():

        seasonal_data =  gen_data[gen_data['season'] == season]

        hum_std = seasonal_data['hum'].std()
        wind_speed_std = seasonal_data['wind_speed'].std()
        t1_std = seasonal_data['t1'].std()
        t2_std = seasonal_data['t2'].std()
        
        for i in gen_data[gen_data['season'] == season].index:
            if np.random.randint(2) == 1:
                gen_data['hum'].values[i] += hum_std/10
            else:
                gen_data['hum'].values[i] -= hum_std/10
                
            if np.random.randint(2) == 1:
                gen_data['wind_speed'].values[i] += wind_speed_std/10
            else:
                gen_data['wind_speed'].values[i] -= wind_speed_std/10
                
            if np.random.randint(2) == 1:
                gen_data['t1'].values[i] += t1_std/10
            else:
                gen_data['t1'].values[i] -= t1_std/10
                
            if np.random.randint(2) == 1:
                gen_data['t2'].values[i] += t2_std/10
            else:
                gen_data['t2'].values[i] -= t2_std/10

	

    return gen_data

def run_model(x_train, x_val, y_train, y_val, cat_vars, num_vars, header):


	#We try to make the data more Gaussianlike with the PowerTransformer. It helps to stabilice variance and mimimiza skewness

	transformer = preprocessing.PowerTransformer()
	y_train = transformer.fit_transform(y_train.values.reshape(-1,1))
	y_val = transformer.transform(y_val.values.reshape(-1,1))


	rang = abs(y_train.max()) + abs(y_train.min())

	num_4_treeModels = pipeline.Pipeline(steps=[
		('imputer', impute.SimpleImputer(strategy='constant', fill_value=-9999))
  											])

	cat_4_treeModels = pipeline.Pipeline(steps=[
		('imputer', impute.SimpleImputer(strategy='constant', fill_value='missing')),
		('ordinal', preprocessing.OrdinalEncoder()) 
	])

	tree_prepro = compose.ColumnTransformer(transformers=[
		('num', num_4_treeModels, num_vars),
		('cat', cat_4_treeModels, cat_vars),
	], remainder='drop') # Drop other vars not specified in num_vars or cat_vars

	tree_classifiers = {
	"Decision Tree": DecisionTreeRegressor(),
	"Extra Trees":   ExtraTreesRegressor(n_estimators=100),
	"Random Forest": RandomForestRegressor(n_estimators=100),
	"AdaBoost":      AdaBoostRegressor(n_estimators=100),
	"Skl GBM":       GradientBoostingRegressor(n_estimators=100),
	"XGBoost":       XGBRegressor(n_estimators=100),
	"LightGBM":      LGBMRegressor(n_estimators=100),
	"CatBoost":      CatBoostRegressor(n_estimators=100, verbose = False),
	}

	tree_classifiers = {name: pipeline.make_pipeline(tree_prepro, model) for name, model in tree_classifiers.items()}

	results = pd.DataFrame({'Model': [], 'MSE': [], 'MAB': [], "Perc. error": [], 'Time': []})

	for model_name, model in tree_classifiers.items():
		
		start_time = time.time()
		model.fit(x_train, y_train)
		total_time = time.time() - start_time
			
		pred = model.predict(x_val)
		
		results = results.append({"Model":    model_name,
								"MSE": metrics.mean_squared_error(y_val, pred),
								"MAB": metrics.mean_absolute_error(y_val, pred),
								"Perc. error": metrics.mean_squared_error(y_val, pred) / rang,
								"Time":     total_time},
								ignore_index=True)


	results_ord = results.sort_values(by=['MSE'], ascending=True, ignore_index=True)
	results_ord.index += 1 
	#results_ord.style.bar(subset=['MSE', 'MAE'], vmin=0, vmax=100, color='#5fba7d')
	print(header)
	print(results_ord)
	
	return results_ord

def main():
	
	#Get data
	data = pd.read_csv(datafile)

	# Data prepareation
 
	data['year'] = data['timestamp'].apply(lambda row: row[:4])
	data['month'] = data['timestamp'].apply(lambda row: row.split('-')[1][:2] )
	data['day'] = data['timestamp'].apply(lambda row: row.split('-')[2][:2] )
	data['hour'] = data['timestamp'].apply(lambda row: row.split(':')[0][-2:] )

	'''
	print(data['year'])
	print(data['month'])
	print(data['hour'])
	'''
	data.drop('timestamp', axis=1, inplace=True)

	y = data['cnt']
	x = data.drop(['cnt'], axis=1)

	#define categorical and numerical variables
	cat_vars = ['season','is_weekend','is_holiday','year','month', 'day','weather_code']
	num_vars = ['t1','t2','hum','wind_speed']

	# Set train and test data
	x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y, test_size=0.2,  random_state=23)

	results0 = run_model(x_train, x_val, y_train, y_val, cat_vars, num_vars, 'No enhancement')
	results0  = results0.assign(type='No enhancement') 
	
	# Run with basic data

	new_data = generate_data(data)
	#We reduce the amount with sampling
	new_data = new_data.sample(new_data.shape[0] // 4)
	
	# We add the new data to the train sample. 
	x_train = pd.concat([x_train, new_data.drop(['cnt'], axis=1 ) ])
	y_train = pd.concat([y_train, new_data['cnt'] ])

	# Run with enhanced data
	results1 = run_model(x_train, x_val, y_train, y_val, cat_vars, num_vars, 'Enhanced')
	results1  = results1.assign(type='Enhanced') 
	
	results = pd.concat([results0, results1])
	results.set_index('Model')
	
	sns.set_style ('darkgrid')


	sns.barplot(x = 'Model', y = 'Perc. error', hue = 'type',  data = results, ci = None)
	plt.show()

 
	return 




if __name__ == '__main__':
	main()