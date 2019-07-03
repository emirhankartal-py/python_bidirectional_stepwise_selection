
# Import Libraries
import BidirectionalStepwiseSelection as ss
import pandas as pd

# Read Data
df = pd.read_csv("test_data.csv")

# Dependent and Independent Variables
X = df.drop(columns= "Exited")
y = df.Exited

# Magic Happens
final_vars, iterations_logs = ss.BidirectionalStepwiseSelection(X,y, model_type="logistic")

# Write Logs To .txt
iterations_file = open("Iterations_logs.txt","w+") 
iterations_file.write(iterations_logs)
iterations_file.close()
