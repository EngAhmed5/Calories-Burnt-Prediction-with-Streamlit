import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler , MinMaxScaler , PolynomialFeatures , OneHotEncoder


def read_data (data_path ):
   #---read data ---
   data = pd.read_csv(data_path)
   #---Drop Useless Column -----
   data = data.drop('User_ID' , axis= 1)
   #---Split Data -----
   x = data.drop('Calories', axis=1)
   y = data['Calories']
   
   print("Data Loaded Successfully")
   return x , y , data


def split_data (x,y) :
   x_train , x_val , y_train , y_val =   train_test_split(x , y , test_size= 0.2 , random_state= 42 ,shuffle= True)
   print ("Data Splited Successfully")
   return x_train , x_val , y_train , y_val




def get_processor(x_train, x_val, num_col, cat_col, proc_num=1):
   
   if proc_num == 1:
      processor = StandardScaler()
   elif proc_num == 2:
      processor = MinMaxScaler()
   else:
      raise ValueError("Invalid Choice. Only 1: StandardScaler or 2: MinMaxScaler")

   # 2. Fit & Transform لـ x_train ثم Transform لـ x_val
   x_train[num_col] = processor.fit_transform(x_train[num_col])
   x_val[num_col] = processor.transform(x_val[num_col])

   # 3. OneHotEncoding لـ categorical columns
   encoder = OneHotEncoder(sparse_output=False
, drop='first')

   encoded_train = encoder.fit_transform(x_train[cat_col])
   encoded_val = encoder.transform(x_val[cat_col])

   encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(cat_col), index=x_train.index)
   encoded_val_df = pd.DataFrame(encoded_val, columns=encoder.get_feature_names_out(cat_col), index=x_val.index)

   # 4. Drop old categorical columns and concat new encoded ones
   x_train = x_train.drop(columns=cat_col)
   x_val = x_val.drop(columns=cat_col)

   x_train = pd.concat([x_train, encoded_train_df], axis=1)
   x_val = pd.concat([x_val, encoded_val_df], axis=1)
   
   print("Data Scaled Successfully")
   
   return x_train, x_val, processor, encoder



