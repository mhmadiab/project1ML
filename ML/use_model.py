import pickle
import numpy as np

#create new local classifier and scaler loaded from the pickle files 
#rb:reading binary

local_classifier= pickle.load(open("classifier.pickle","rb"))
local_scalar= pickle.load(open("sc.pickle","rb"))

#new prediction using the new local classifier and scaler
new_pred= local_classifier.predict(local_scalar.transform(np.array([[23,3000]])))
