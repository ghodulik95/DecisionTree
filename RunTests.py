import subprocess
from collections import OrderedDict

#Use an ordered dictionary so we can perform the quicker tests first
tests = OrderedDict()
#Add (testName, command) key-value pairs
tests['voting-dept-1']=['python', 
			r'C:\\Users\\George\\Documents\\VMSharedFolder\\EECS440PA1\\code\\code\\python\\main.py',
			'--dataset_directory',
			'data',
			'--dataset',
			'voting',
			'dtree',
			'--depth',
			'1']
tests['voting-dept-inf']=['python', 
			r'C:\\Users\\George\\Documents\\VMSharedFolder\\EECS440PA1\\code\\code\\python\\main.py',
			'--dataset_directory',
			'data',
			'--dataset',
			'voting',
			'dtree',
			'--depth',
			'0']
tests['spam-dept-1']=['python', 
			r'C:\\Users\\George\\Documents\\VMSharedFolder\\EECS440PA1\\code\\code\\python\\main.py',
			'--dataset_directory',
			'data',
			'--dataset',
			'spam',
			'dtree',
			'--depth',
			'1']
tests['volcanoes-dept-1']=['python', 
			r'C:\\Users\\George\\Documents\\VMSharedFolder\\EECS440PA1\\code\\code\\python\\main.py',
			'--dataset_directory',
			'data',
			'--dataset',
			'volcanoes',
			'dtree',
			'--depth',
			'1']
tests['spam-dept-2']=['python', 
			r'C:\\Users\\George\\Documents\\VMSharedFolder\\EECS440PA1\\code\\code\\python\\main.py',
			'--dataset_directory',
			'data',
			'--dataset',
			'spam',
			'dtree',
			'--depth',
			'2']
tests['volcanoes-dept-2']=['python', 
			r'C:\\Users\\George\\Documents\\VMSharedFolder\\EECS440PA1\\code\\code\\python\\main.py',
			'--dataset_directory',
			'data',
			'--dataset',
			'volcanoes',
			'dtree',
			'--depth',
			'2']
tests['spam-dept-3']=['python', 
			r'C:\\Users\\George\\Documents\\VMSharedFolder\\EECS440PA1\\code\\code\\python\\main.py',
			'--dataset_directory',
			'data',
			'--dataset',
			'spam',
			'dtree',
			'--depth',
			'3']
tests['volcanoes-dept-3']=['python', 
			r'C:\\Users\\George\\Documents\\VMSharedFolder\\EECS440PA1\\code\\code\\python\\main.py',
			'--dataset_directory',
			'data',
			'--dataset',
			'volcanoes',
			'dtree',
			'--depth',
			'3']
tests['spam-dept-4']=['python', 
			r'C:\\Users\\George\\Documents\\VMSharedFolder\\EECS440PA1\\code\\code\\python\\main.py',
			'--dataset_directory',
			'data',
			'--dataset',
			'spam',
			'dtree',
			'--depth',
			'4']
tests['volcanoes-dept-4']=['python', 
			r'C:\\Users\\George\\Documents\\VMSharedFolder\\EECS440PA1\\code\\code\\python\\main.py',
			'--dataset_directory',
			'data',
			'--dataset',
			'volcanoes',
			'dtree',
			'--depth',
			'4']
tests['spam-dept-5']=['python', 
			r'C:\\Users\\George\\Documents\\VMSharedFolder\\EECS440PA1\\code\\code\\python\\main.py',
			'--dataset_directory',
			'data',
			'--dataset',
			'spam',
			'dtree',
			'--depth',
			'5']
tests['volcanoes-dept-5']=['python', 
			r'C:\\Users\\George\\Documents\\VMSharedFolder\\EECS440PA1\\code\\code\\python\\main.py',
			'--dataset_directory',
			'data',
			'--dataset',
			'volcanoes',
			'dtree',
			'--depth',
			'5']

#For each test
for fileName, args in tests.iteritems():
	#Run the tests and capture its printouts
	result = subprocess.check_output(args)
	#Save the printouts to a file
	f = open(r'C:\\Users\\George\\Documents\\VMSharedFolder\\EECS440PA1\\code\\code\\python\\tests\\'+fileName+'.txt', 'w')
	f.write(result)
	f.close()
	#Print out that this test has finished
	print fileName+' test is done '
