from sklearn.metrics import mean_squared_error , r2_score
import numpy as np 



def evaluate_model (model , x , y  , model_name , label ):
   
   print (f"Evaluate {model_name} Model On {label} ")
   print("-"*60)
   #----Prediction ------------
   y_pred =  model.predict(x)
   #------Calculate Error -----------
   MSE = mean_squared_error(y , y_pred)
   print (f"MSE for {model_name} Model On {label} Data : {MSE:.2f} ")
   RMSE = np.sqrt(MSE)
   print (f"RMSE for {model_name} Model On {label} Data : {RMSE:.2f} ")
   r2 = r2_score(y , y_pred)
   print (f"R2-Score for {model_name} Model On {label} Data :{r2 * 100:.2f}%")
   
   return MSE , RMSE , r2
   
