import numpy
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import csv

from tslearn import metrics
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler


# Check validity of clusters, cluster score
# Find optimal number of clusters

papers = []
timeSeriesDataset = []
maxSelect = 1000

with open('dataset.csv', mode='r') as dataFile:
    index = 0
    csvFile = csv.reader(dataFile)
    count = 0
    for line in csvFile:
        # md = [line[0], line[1]]
        # papers.append(md)

        thisPaper = [float(freq) for freq in line[2:]].copy()
        timeSeriesDataset.append(thisPaper)

        thisPaper.insert(0, line[1])
        thisPaper.insert(0, line[0])
        papers.append(thisPaper)

        count += 1
        if count>maxSelect:
            break

X_train = to_time_series_dataset(timeSeriesDataset)
# # X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
# X_train = TimeSeriesResampler(sz=10).fit_transform(X_train)
# sz = X_train.shape[1]

print("Data In")
print("Processing...")
km = TimeSeriesKMeans(n_clusters=7, metric="dtw")
labels = km.fit_predict(X_train)
print("Processed")

for i in range(0, len(papers)):
    papers[i].insert(0, labels[i])

with open('output.csv', 'w', newline='') as f: 
        write = csv.writer(f) 
        write.writerows(papers) 


