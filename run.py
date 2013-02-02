import sys

""" 
This is the entry point of the application
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

	print testData

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

	print trainData	



if __name__ == "__main__":

	fTrainIn = get_training_file()
	fTestIn = get_test_file()
	run_app(fTrainIn,fTestIn)
	



