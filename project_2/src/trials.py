

import numpy as np
import pandas

# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([0, 0, 1, 1])
#
# from sklearn.svm import SVC
#
# clf = SVC(kernel='sigmoid', gamma='auto', probability=True)
# # clf = SVC(kernel='linear', gamma='auto', probability=True)
# # clf = SVC(kernel='poly', gamma='auto', probability=True)
# # clf = SVC(kernel='rbf', gamma='auto', probability=True)
#
# clf.fit(X, y)
#
# print(clf.score([[-1.1, -1], [1.5, 1]], pandas.Series([0, 1])))
#
# print(clf.predict_proba([[-0.8, -1]]))

a = [1,2,3,4,5,6,7,7]

b = {"df": a,
    "df1": a,
    "df2": a,
    "df3": a}

import json

with open("/Users/andreidm/Desktop/test.json", 'w') as file:
    json.dump(b, file)

# with open("/Users/andreidm/Desktop/test.txt", 'a') as file:
#     file.write(a.__str__())
#     file.write(b.__str__())
