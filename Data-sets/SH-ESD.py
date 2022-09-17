import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib as plt
import matplotlib.patches as mpatches
from sklearn import svm
import sesd


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


# ------------ SH-ESD : avg_framerate -------------------- # 
print("SH-ESD : avg_framerate") 
df = pd.DataFrame(attack4["avg_framerate"])
times = list(df["avg_framerate"])
print(times)
outliers_indices = sesd.seasonal_esd(times,hybrid = True, max_anomalies=10)
print(outliers_indices)
for idx in outliers_indices:
    print(f'Anomaly index: {idx}, anomaly value: {times[idx]}')
print("---------------------------------------------------------")
print()
print()


# ------------ SH-ESD : Avg_Frametime -------------------- # 
print("SH-ESD : Avg_Frametime") 
df = pd.DataFrame(attack4["avg_frametime"])
times = list(df["avg_frametime"])
#print(list)
outliers_indices = sesd.seasonal_esd(times,hybrid = True, max_anomalies=10)
print(outliers_indices)
for idx in outliers_indices:
    print(f'Anomaly index: {idx}, anomaly value: {times[idx]}')

print("---------------------------------------------------------")
print()
print()

# ------------ SH-ESD : stdev_frametime -------------------- # 
print("SH-ESD : stdev_frametime") 
df = pd.DataFrame(attack4["stdev_frametime"])
times = list(df["stdev_frametime"])
#print(list)
outliers_indices = sesd.seasonal_esd(times,hybrid = True, max_anomalies=10)
print(outliers_indices)
for idx in outliers_indices:
    print(f'Anomaly index: {idx}, anomaly value: {times[idx]}')
print("---------------------------------------------------------")
print()
print()
# ------------ SH-ESD : entropy_framerate -------------------- # 
print("SH-ESD : entropy_framerate") 
df = pd.DataFrame(attack4["entropy_framerate"])
times = list(df["entropy_framerate"])
#print(list)
outliers_indices = sesd.seasonal_esd(times,hybrid = True, max_anomalies=10)
print(outliers_indices)
for idx in outliers_indices:
    print(f'Anomaly index: {idx}, anomaly value: {times[idx]}')
print("---------------------------------------------------------")
print()
print()

# ------------ SH-ESD : AppMotionToPhotonLatency -------------------- # 
print("SH-ESD : AppMotionToPhotonLatency") 
df = pd.DataFrame(attack4["AppMotionToPhotonLatency"])
times = list(df["AppMotionToPhotonLatency"])
#print(list)
outliers_indices = sesd.seasonal_esd(times,hybrid = True, max_anomalies=10)
print(outliers_indices)
for idx in outliers_indices:
    print(f'Anomaly index: {idx}, anomaly value: {times[idx]}')
print("---------------------------------------------------------")
print()
print() 


# ------------ SH-ESD : Control Group -------------------- # 
print("SH-ESD : Control Group") 
ts = np.random.random(100)
# Introduce artificial anomalies
ts[14] = 9
ts[83] = 10
outliers_indices = sesd.seasonal_esd(ts, hybrid=True, max_anomalies=10)
for idx in outliers_indices:
    print(f'Anomaly index: {idx}, anomaly value: {ts[idx]}')
print("---------------------------------------------------------")
print()
print()




