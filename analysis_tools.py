# analysis_tools.py
import numpy as np
import matplotlib.pyplot as plt


class ModelAnalysis:
	def plot_hyperparameter(parameter_list,train_mae_list,val_mae_list, hyperparam_name="Hyperparameter"):
	
		plt.figure(figsize=(8,5))
		plt.plot(parameter_list, train_mae_list, marker='o', label='Training MAE')
		plt.plot(parameter_list, val_mae_list, marker='s', label='Validation MAE')
		plt.title(f"MAE vs {hyperparam_name}")
		plt.xlabel(hyperparam_name)
		plt.ylabel("MAE")
		plt.legend()
		plt.grid(True)
		plt.show()
		
