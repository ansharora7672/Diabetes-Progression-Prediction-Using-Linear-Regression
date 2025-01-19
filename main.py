import matplotlib.pyplot as plt
import numpy as np 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# scikit learn has a already prebuilt dataset so I am using that rigght now just to learn adn get strated with regression algorithm diabetes dataset is the name.....

diabetes = datasets.load_diabetes()

#print(diabetes.keys())

# after running the above line of the code we get the following output - 
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])

#dataset ki slicing hori h idhr
#diabetes_x = diabetes.data
#print(diabetes.DESCR)

# this line below is shaping the diabetes dataset to use only one feature that is 3rd column 
#this is working witht eh data component of the diabetes dataeswet whcih si th BMI
# np.newaxis: Adds a new axis, converting the data into a shape suitable for the model.

#modifying hte code with multiple . chanign the line 22 to
# isnetad of this  diabetes_X = diabetes.data[:,np.newaxis, 2]\

diabetes_X = diabetes.data

# the BMI values appear as small decimals, they're correctly processed for use in your regression model. and it is a 2 d array,. iit madew an array and in that put al the things in the array.
# print(diabetes_X)

#training the model..gave the last 30 eelemtns for trianing
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

#y data
diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]

#using hte linear regression model now whihc is there in the scikit.
model = linear_model.LinearRegression()

# now i will makew a line . so for that i am fitting the data into the model. the modewl will make the line for me .....
# this is how we feed hte data to the model.......
model.fit(diabetes_X_train, diabetes_Y_train)

# now we will add values to teh line and see the predicition. testing 
diabetes_Y_predict = model.predict(diabetes_X_test) 
#print(diabetes_Y_predict)
#print("line gap gicing to see the acutal y values tooooooo...")
#print(diabetes_Y_test)


# till this stage this was only donw ith only one feature.....

#before moviong on we will be looking at hte values whihc we got fromt he above out .. how valid accurate they are . for this the best thing we can do is calcuate the mean error.....woof we see if tha tis too much we know that the model isnt that good. but as of now the mean ror would be a big one bcs there are only 1 feature which we aree using to determine.
print("mean squared error is ", mean_squared_error(diabetes_Y_test, diabetes_Y_predict))


#weights are the featuers . we can have multiple wirghts . but at this point of the projecct . we are just ocnsidering only one fetuare which is the bmi that is the 3rd column int hte data .
print("Weights - ", model.coef_)
print("intercept/slope - ", model.intercept_)

# now plotting . only with the one feature. 

#plt.scatter(diabetes_X_test, diabetes_Y_test)
#plt.plot(diabetes_X_test, diabetes_Y_predict)
#plt.show()

#modifying hte code with multiple . chanign the line 22 to
# isnetad of this  diabetes_X = diabetes.data[:,np.newaxis, 2]\
# we will do diabetes_X = diabetes.datawe have to comment 61, 62, 63 to make thsi working with all weights . we cnat plot


# i want to give it a single data to predict now 
new_data_point = np.array([[0.04, 0.05, 0.06, 0.02, 0.08, 0.03, 0.07, 0.01, 0.09, 0.04]])
new_prediction = model.predict(new_data_point)
print("Prediction for new data point:", new_prediction)

print(diabetes_X)
print(diabetes.feature_names)

