from sklearn import datasets
import numpy as np
import pandas as pd
data=datasets.load_diabetes()
#print(data) isse got info that input ko data bol aur output ko target
input=data.data
df=pd.DataFrame(input)
#print(df)
output=data.target
df.columns=data.feature_names #column names change kar diya
#print(df)
from sklearn import model_selection
train_input,test_input,train_output,test_output=model_selection.train_test_split(input,output)
#now import linear regression and fit data in it
from sklearn.linear_model import LinearRegression
algorithm=LinearRegression()
algorithm.fit(train_input,train_output)
output_predicted=algorithm.predict(test_input)
import matplotlib.pyplot as plt
x=np.arange(0,400,0.1)
plt.plot(x,x,color="blue")
plt.scatter(test_output,output_predicted,color="red")
plt.axis([0,400,0,400])
plt.show()
#clearly output predicted matches as y=x ke kareeb very well with test output which was actual output