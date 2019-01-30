import pandas as pd
import pickle
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer

def download_datasets():

	i = 0

	i += 1
	print("%d of 10" %i)
	print("Iris\n")
	iris = load_iris()
	X1 = iris.data
	Y1 = iris.target

	i += 1
	print("%d of 10" %i)
	print("Digits\n")
	digits = load_digits()
	X2 = digits.data
	Y2 = digits.target

	i += 1
	print("%d of 10" %i)
	print("Breast cancer\n")
	breast_cancer = load_breast_cancer()
	X3 = breast_cancer.data
	Y3 = breast_cancer.target

	i += 1
	print("%d of 10" %i)
	print("Sensorless drive diagnosis\n")
	Sensorless_data = pd.read_csv(
	'https://archive.ics.uci.edu/ml/machine-learning-databases/00325/Sensorless_drive_diagnosis.txt',
							   sep= ' ', header= None)
	X4 = Sensorless_data.values[:, 0:48]
	Y4 = Sensorless_data.values[:,48]

	i += 1
	print("%d of 10" %i)
	print("Banknote authentication\n")
	banknote_data = pd.read_csv(
	'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt',
							   sep= ',', header= None)
	X5 = banknote_data.values[:,0:4]
	Y5 = banknote_data.values[:,4]

	i += 1
	print("%d of 10" %i)
	print("Balance\n")
	balance_data = pd.read_csv(
	'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
							   sep= ',', header= None)
	X6 = balance_data.values[:,1:5]
	Y6 = balance_data.values[:,0]

	i += 1
	print("%d of 10" %i)
	print("Wifi localization\n")
	wifi_data = pd.read_csv(
	'https://archive.ics.uci.edu/ml/machine-learning-databases/00422/wifi_localization.txt',
							   sep= '\s+', header= None)
	X7 = wifi_data.values[:,0:7]
	Y7 = wifi_data.values[:,7]

	i += 1
	print("%d of 10" %i)
	print("CMC\n")   
	cmc_data = pd.read_csv(
	'https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data',
							   sep= ',', header= None)
	X8 = cmc_data.values[:,0:9]
	Y8 = cmc_data.values[:,9]

	i += 1
	print("%d of 10" %i)
	print("Yeast\n")
	yeast_data = pd.read_csv(
	'https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data',
							   sep= '\s+', header= None)
	X9 = yeast_data.values[:,1:9]
	Y9 = yeast_data.values[:,9]

	i += 1
	print("%d of 10" %i)
	print("Abalone\n")
	abalone_data = pd.read_csv(
	'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data',
							   sep= ',', header= None)
	X10 = abalone_data.values[:,0:8]
	for i in X10:
		if i[0] == 'M':
			i[0] = 1
		elif i[0] == 'F':
			i[0] = 2
		else:
			i[0] = 3   
	Y10 = abalone_data.values[:,8].astype(int)

	print("Saving the datasets for future use.")
	pickle.dump((X1, X2, X3, X4, X5, X6, X7, X8, X9, X10), open('datasets_x.pkl', 'wb'))
	pickle.dump((Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10), open('datasets_y.pkl', 'wb'))

download_datasets()