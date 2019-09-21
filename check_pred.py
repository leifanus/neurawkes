# Check the properties of prediction
import os
import pandas as pd 
import numpy as np 
import collections
import matplotlib.pyplot as plt

def HistTime(dtime_list):
	'''
	Draw the bar plot of the duration time 
	: params: dtime_list: duration time list
	'''
	# load dataset
	count = collections.defaultdict()
	threshold = [10**(i) for i in range(-6, 2)]

	for i in range(len(threshold)):
		count[i] = 0.0

	for dtime in dtime_list:
		for i in range(len(threshold)):
			if dtime < threshold[i]:
				count[i] += 1

	x = []
	y = []
	for i in range(len(threshold)):
		x.append(i)
		print(i, count[i])
		if i==0:
			y.append(count[i]/len(dtime_list))
		else:
			y.append((count[i] - count[i-1])/len(dtime_list))

	# print(x, y)
	plt.title('Label arrival time')
	plt.xlabel('dtime')
	plt.ylabel('density')
	plt.bar(x, y)
	plt.xticks(x, threshold)
	# plt.show()

def HistType(type_list, num_types):
	'''
	Draw the bar plot for the label types.
	: params: type_list: list of event types
	'''

	counter = collections.Counter(type_list)
	for i in range(num_types):
		if i not in counter:
			counter[i] = 0.0
	print(counter)
	x = [key for key in sorted(counter)]
	y = [counter[key] for key in sorted(counter)]
	y /= np.sum(y, dtype=np.float)
	plt.title('Label types')
	plt.xlabel('types')
	plt.ylabel('density')
	plt.bar(x, y)
	# plt.show()

########################################################################

file_names = os.listdir('./tmp/D1/')

time_list = []

for names in file_names:
	time = names[10:]
	if time is not '' and time not in time_list:
		time_list.append(time)

time_pred = []
type_pred = []
num_types = 3

for time in time_list:
	time_file = pd.read_csv('./tmp/D1/' + 'time_pred_' + time)
	time_file = np.array(time_file)[:, 1:]
	type_file = pd.read_csv('./tmp/D1/' + 'type_pred_' + time)
	type_file = np.array(type_file)[:, 1:]
	time_pred.extend(time_file[-1, :])
	type_pred.extend(type_file[-1, :])

# HistTime(time_pred)
# HistType(type_pred, num_types)
plt.subplot(121)
HistTime(time_pred)
plt.subplot(122)
HistType(type_pred, num_types)
plt.show()
