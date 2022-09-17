import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib as plt
from sklearn import svm


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

u_data = pd.read_csv('Data-sets\BlessingReworked-TestWithThisNormalAndAttack1.csv')
attack1 = pd.read_csv('Data-sets\BlessingReworked-TestWithThisNormalAndAttack1.csv')
attack2 = pd.read_csv('Data-sets\BlessingReworked-TestWithThisNormalAndAttack2.csv')
attack3 = pd.read_csv('Data-sets\BlessingReworked-TestWithThisNormalAndAttack3.csv')
attack4 = pd.read_csv('Data-sets\BlessingReworked-TestWithThisNormalAndAttack4.csv')


# print(attack1)
#trainx = u_data[['avg_framerate','avg_frametime','max_frametime','min_frametime','stdev_frametime','entropy_framerate']]

trainy = u_data['label']
trainx = u_data[['avg_framerate', 'avg_frametime', 'stdev_frametime', 'entropy_framerate', 'AppMotionToPhotonLatency']]

testy1 = attack1['label']
testx1 = attack1[['avg_framerate', 'avg_frametime', 'stdev_frametime', 'entropy_framerate', 'AppMotionToPhotonLatency']]

testy2 = attack2['label']
testx2 = attack2[['avg_framerate', 'avg_frametime', 'stdev_frametime', 'entropy_framerate', 'AppMotionToPhotonLatency']]

testy3 = attack3['label']
testx3 = attack3[['avg_framerate', 'avg_frametime', 'stdev_frametime', 'entropy_framerate', 'AppMotionToPhotonLatency']]

testy4 = attack4['label']
testx4 = attack4[['avg_framerate', 'avg_frametime', 'stdev_frametime', 'entropy_framerate', 'AppMotionToPhotonLatency']]

testx = testx4
testy = testy4

scaler = StandardScaler()
trainx = scaler.fit_transform(trainx)
testx = scaler.fit_transform(testx)

print(trainx)

# training the model
lregressor = svm.SVC()
lregressor.fit(trainx,trainy)

# predictions
lregressor_result = lregressor.predict(testx)
result = []

# This was reversed from Isolation Forest?
for i in lregressor_result:
    if i == 1:
        result.append(1)
    else:
        result.append(0)

result = pd.Series(result)
results = np.column_stack((testy, result))

test_state = []
class_state = []
for i in testy:
    if i == 0:
        state = "Current state: No Attack"
        test_state.append(state)
    else:
        state = "Current state: Attack!!!"
        test_state.append(state)

for i in result:
    if i == 0:
        state = "AI Classification: No Attack"
        class_state.append(state)
    else:
        state = "AI Classification: Attack!!!"
        class_state.append(state)

res = "\n".join("{} {}".format(x, y) for x, y in zip(test_state, class_state))
print(res)
print("SVM Accuracy:", metrics.accuracy_score(testy, result))
df = pd.DataFrame(testy)
df2 = pd.DataFrame(result)
df.to_csv('testy.csv')
df2.to_csv('result.csv')




