import sys
from dtree import *

""" 
This is the entry point of the application. The main funtion that calls the 
create decision tree algorithm is run_app. This in turn is called from main

To run
=======
python -u run.py train.dat test.dat

Data Strucutre
---------------

The funtion: create_decision_tree(examples, attributes, target_attribute, heuristic_funtion)
takes in the following input:

examples (train or test data) : list of dicts (python dictionaries)
attributes : list
target_attribute: string
heuristic_funtion: funtion pointer to "gain" funtion
"""





def get_training_file():
	"""
	Tries to extract the first argument from the command line. Else asks the user to
	enter the training data filename
	"""

	if len(sys.argv) < 3:
		# Print the prompt for training file name
		print "Please enter the training file: ",
		training_filename = sys.stdin.readline().strip()

	else:
		training_filename = sys.argv[1]


	try:
		fTrainIn = open(training_filename,"r")
	except IOError:
		print "Error: Could not find the training file specified or unable to open it" %training_filename
		sys.exit(0)

	return fTrainIn

def get_test_file():
	"""
	Tries to extract the second argument from the command line. Else asks the user to
	enter the test data filename
	"""

	if len(sys.argv) < 3:
		# Print the prompt for test file name
		print "Please enter the test file: ",
		test_filename = sys.stdin.readline().strip()

	else:
		test_filename = sys.argv[2]


	try:
		fTestIn = open(test_filename,"r")
	except IOError:
		print "Error: Could not find the test file specified or unable to open it" %test_filename
		sys.exit(0)

	return fTestIn

def prepare_attributes(attrList):
	"""
	Returns a  list of attributes with the sizes removed and also returns a 
	dict item with the attributes as key and the number
	"""

	attrList = attrList[:]
	attrs = []
	attrsDict = {}

	for i in xrange(0, len(attrList)-1, 2):
		attrs.append(attrList[i])
		# set the value of attribute name as key to its number
		attrsDict[attrList[i]] = attrList[i+1] 

	return attrs, attrsDict



def run_app(fTrainIn, fTestIn):
	"""
	Runs the algorithm on the data
	"""
	linesInTest = [line.strip() for line in fTestIn.readlines()]
	attributes = linesInTest[0].split(" ")
	targetAttribute = "class"
	#once we have the attributes remove it from lines
	linesInTest.reverse()
	linesInTest.pop() # pops from end of list, hence the two reverses
	linesInTest.reverse()		

	attrList, attrDict = prepare_attributes(attributes)
	attrList.append(targetAttribute)
	#print attrList
	attrDict[targetAttribute] = 2 # since its a binary classification 

	# prepare data
	testData = []
	for line in linesInTest:
		testData.append(dict(zip(attrList,[datum.strip() for datum in line.split("\t")])))

	#print testData

	linesInTrain = [lineTrain.strip() for lineTrain in fTrainIn.readlines()]
	attributesTrain = linesInTrain[0].replace("\t"," ").split(" ")
	#print attributesTrain
	targetAttributeTrain = "TrainClass"
	#once we have the attributes remove it from lines
	linesInTrain.reverse()
	linesInTrain.pop() # pops from end of list, hence the two reverses
	linesInTrain.reverse()		

	attrListTrain, attrDictTrain = prepare_attributes(attributesTrain)
	attrListTrain.append(targetAttributeTrain)
	#print attrListTrain
	attrDictTrain[targetAttributeTrain] = 2 # since its a binary classification 

	# prepare data
	trainData = []
	for lineTrain in linesInTrain:
		trainData.append(dict(zip(attrListTrain,[datum.strip() for datum in lineTrain.split("\t")])))


	trainingTree = create_decision_tree(trainData, attrListTrain, targetAttributeTrain, gain)
	trainingClassification = classify(trainingTree, trainData)

	testTree = create_decision_tree(testData, attrList, targetAttribute, gain)
	testClassification = classify(testTree, testData)

	# also returning the example classification in both the files
	givenTestClassification = []
	for row in testData:
		givenTestClassification.append(row[targetAttribute])

	givenTrainClassification = []
	for row in trainData:
		givenTrainClassification.append(row[targetAttributeTrain])

	return trainingTree, trainingClassification, testClassification, givenTrainClassification, givenTestClassification


def accuracy(algoclassification, targetClassification):
	matching_count = 0.0

	for alg, target in zip(algoclassification,targetClassification):
		if alg == target:
			matching_count += 1.0

	#print len(algoclassification)
	#print len(targetClassification)
	return (matching_count / len(targetClassification)) * 100

def print_tree(tree, str):
	"""
	Funtion to print the treee in the desired format
	"""

	if type(tree)== dict:
		#print "%s%s = " % (str,tree.keys()[0]),
		for item in tree.values()[0].keys():
			print "%s%s = %s" % (str, tree.keys()[0],item),
			print " : "
			print "|",
			print_tree(tree.values()[0][item],str + "  ")
	else:
		print "%s : %s" % (str, tree)

if __name__ == "__main__":

	fTrainIn = get_training_file()
	fTestIn = get_test_file()
	trainingTree, trainingClassification, testClassification, givenTrainClassification, givenTestClassification = run_app(fTrainIn,fTestIn)
	print_tree(trainingTree,"")
	print " Accuracy of training set (%s instances) : " % len(givenTrainClassification),
	print accuracy(trainingClassification, givenTrainClassification)
	print " Accuracy of test set (%s instances) : " % len(givenTestClassification),
	print accuracy(testClassification, givenTestClassification)




