from Data_processing import * 
from modeling import * 
from Evaluate import *


if __name__ == '__main__' :
   #----Data Path-----
   data_path = r"D:\Ahmed\Projects\Calories Burnt Prediction\Data\data.csv"
   #---Read Data------
   x,y,data = read_data(data_path)
   print("="*60)
   #----Split Data-----
   x_train , x_val , y_train , y_val = split_data(x , y)
   print("="*60)
   #----Scale Data-----
   
   num_col =  ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate','Body_Temp']
   cat_col = ['Gender']
   
   try :
      num = int(input("Choose Scaler To Scale The Data\n (1) StandardScaler\n (2) MinMaxScaler\n Your Choice: "))
      if num in [1,2] :
         x_train , x_val , prosessor , encoder = get_processor(x_train ,x_val ,num_col , cat_col ,num)
   except:
      raise ValueError("Choose From 1 or 2")
   
   print(x_train.columns.to_list())
   print("="*60)
   
   #------Select Model And Evaluate ----------
   try:
      model_num = int(input("Choose Model To Train:\n"
      "(1) Linear Regression\n"
      "(2) Ridge Regression\n"
      "(3) Random Forest\n"
      "(4) Support Vector Machine\n"
      "(5) XGBoost\n"
      "Your Choice: "))
      if model_num in [1,2,3,4,5] :
         trained_model , model_name = model_selection(x_train , y_train ,model_num )
   except:
      raise ValueError("Invalid input. Please enter a number between 1 and 5.")
   
   print("="*60)
   #---Train----
   _,_,_ = evaluate_model(trained_model , x_train , y_train , model_name , label="Train")
   
   print("="*60)
   
   #---Validation---
   _,_ ,_= evaluate_model(trained_model , x_val , y_val , model_name , label="Validation")
   
   #---Save Model ----
   save_model(trained_model , model_name ,scaler=prosessor , encoder=encoder)
   
