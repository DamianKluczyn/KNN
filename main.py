import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import linear_model, preprocessing

#import data
data = pd.read_csv("car.data")

#transform non-integers into integers values (return numpy array)
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
clas = le.fit_transform(list(data["class"]))
safety = le.fit_transform(list(data["safety"]))

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(clas)

#split train/test data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)

print(accuracy)
#0.9653179190751445

predict = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]
"""
for x in range(len(predict)):
    print("Predicted: ", names[predict[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print(n)
"""