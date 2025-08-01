from sklearn.linear_model import LinearRegression , Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error , r2_score
import numpy as np
import joblib
import os 


def model_selection (x_train,y_train, num = 1):
   
   if num == 1 :
      model = LinearRegression() 
      model_name = "Linear Regression"
   elif num == 2 :
      model = Ridge(alpha= 0.01) 
      model_name = "Ridge"
   elif num == 3 :
      model = RandomForestRegressor(n_estimators=200)
      model_name = "Random Forest Regressor"
   elif num == 4 :
      model = SVR(C= 10 , kernel='rbf')
      model_name = "SVR"
   elif num == 5 :
      model = XGBRegressor(learning_rate = 0.1 , max_depth = 5 , n_estimators = 200)
      model_name = "XGB Regressor"
   else:
      raise ValueError("Invalid Choice. Choose a valid model number (1-5).")
   
   #--- Train Model ---
   model.fit(x_train,y_train)
   print(f"{model_name} Trained Successfully")
   return model , model_name



def save_model(model, model_name="model", scaler=None, encoder=None):
   current_dir = os.path.dirname(os.path.abspath(__file__))
   base_dir = os.path.dirname(current_dir)
   save_dir = os.path.join(base_dir, "Saved Models")
   os.makedirs(save_dir, exist_ok=True)

   filename = model_name + ".pkl"
   filepath = os.path.join(save_dir, filename)

   # Bundle all objects in a dictionary
   model_bundle = {
      "model": model,
      "scaler": scaler,
      "encoder": encoder
   }

   joblib.dump(model_bundle, filepath)
   print(f"Model bundle saved as '{filename}' in: {save_dir}")


def load_model(model_name):
   current_dir = os.path.dirname(os.path.abspath(__file__))
   base_dir = os.path.dirname(current_dir)
   save_dir = os.path.join(base_dir, "Saved Models")

   filename = model_name + ".pkl"
   filepath = os.path.join(save_dir, filename)

   if not os.path.exists(filepath):
      raise FileNotFoundError(f"Model '{filename}' not found in {save_dir}")

   # Load the bundle (model + scaler + encoder)
   model_bundle = joblib.load(filepath)
   print(f"Model bundle '{filename}' loaded successfully from: {save_dir}")
   return model_bundle["model"], model_bundle["scaler"], model_bundle["encoder"]
