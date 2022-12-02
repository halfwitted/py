pgm1

l1 = [1, 2, 3, 4, 5]
l2 = [1, 2, 3]
l = len(l2)
count = 0
for i in l1:
    for j in l2:
        if i == j:
            count += 1
if count == l:
    print("True")
else:
    print("False")
    
--------------------------------------------------------------------------------
pgm 2: det

det = [{"name": "joon", "m1": 33, "m2": 9},{"name":"joy","m1":99,'m2':88}]
for i in det:
    m1=i.pop('m1')
    m2=i.pop('m2')
    i["avg"]=(m1+m2)/2
print(det)

--------------------------------------------------------------------------------

pgm3:tup

a=(1,2,3,4,5)
print(a[:2])
print(a[1:4])
print(a[::-1])
b=tuple("autumn")
print(b[:])
print(b[:-1])

--------------------------------------------------------------------------------

pgm 4: numpy

import numpy as np
a=np.array([[1,2],[2,3]])
b=np.array([[1,3],[1,2]])
print("add", a+b)
print("div", np.divide(a,b))

--------------------------------------------------------------------------------

pgm 5: pandas
import pandas as pd
val={"values":[1,2,3,4,5,6,76,78,89,90]}
df=pd.DataFrame(val)
df['square']=df["values"]**2
print(df)

-------------------------------------------------------------------------------- 
pgm 6:pangram

text=input()
l=list(set(text))
if " " in l:
    l.remove(" ")
if len(l)==26:
    print("is a panagram")
else:
    print("not a panagram")
print(l)

--------------------------------------------------------------------------------

pgm 7:KFold
install numpy ,sklearn

from numpy import array
from sklearn.model_selection import KFold
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7])
kfold = KFold(5)

for train, test in kfold.split(data):
	print("train: %s, test: %s" % (data[train], data[test]))

--------------------------------------------------------------------------------
pgm8: csv

import csv
header=['Name','Age'];
data=[["joon",0],["joy",100]]
file="rec.csv"
with open(file,"w") as r:
    csvwriter=csv.writer(r);
    csvwriter.writerow(header)
    csvwriter.writerows(data)

import csv
with open("rec.csv","r") as r:
    rows=csv.reader(r)
    for row in rows:
        print(row)

--------------------------------------------------------------------------------

pgm9: kmeans
install matplotlib

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys

x=[4,5,10,4,3,22,2,84]
y=[21,19,24,16,25,24,22,21]
plt.scatter(x,y)
plt.show()
d=list(zip(x,y))
kmeans = KMeans(n_clusters=2)
kmeans.fit(d)

plt.scatter(x,y,c=kmeans.labels_)
plt.show()
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()

--------------------------------------------------------------------------------

pgm 10: naivebayers
install pandas, sklearn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


df = pd.read_csv("C:\dataset.csv")
x = df.drop("diabetes",axis=1)
y = df["diabetes"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
model = GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)*100

print(accuracy)

--------------------------------------------------------------------------------

pgm11: logistic Regression
install matplotlib, pandas, sklearn

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('c:/dataset.csv')
x = data[['glucose', 'diabetes']]
y = data['bloodpressure']
model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(x,y)
z=model.predict(x)
print(z)
plt.plot(z)
plt.show()

-------------------------------------------------------------------------------- 

pgm 12: stack generation
install sklearn

import sklearn.datasets as sd
import sklearn.model_selection as al
import sklearn.ensemble as se
import sklearn.linear_model as sl
import sklearn.svm as sv
import time
X, Y = sd.load_diabetes(return_X_y=True)
X_train, X_test, Y_train, Y_test = al.train_test_split(X,Y,random_state=42)
stacked = se.StackingRegressor( estimators =[('SVR', sv.SVR()),('Liner',sl.LinearRegression())])
st = time.time()
stacked.fit(X_train, Y_train)
et = time.time()
print("Coefficient of determination: {}".format(stacked.score(X_test, Y_test)))
print("Computation Time: {}".format(et - st))
--------------------------------------------------------------------------------

pgm13: svm
install matplotlib, pandas, numpy, sklearn

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
data = pd.read_csv('c:/dataset.csv')
x = data['glucose']
y = data['diabetes']
tr_x = np.vstack((x, y)).T
tr_y = data['bloodpressure']
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(tr_x,tr_y)
w =clf.coef_[0]
a = -w[0] /w[1]
xx = np.linspace(0, 13)
yy = a* xx -clf.intercept_[0] / w[1]
plt.plot(xx,yy,'k-')
plt.scatter(tr_x[:,0],tr_x[:,1],c=tr_y)
plt.show()
