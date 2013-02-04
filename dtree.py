"""
Functions to build the decision tree including the ID3 heuristic
"""

import math

def entropy(data, targetAttr):
	""" 
	Given the target attribute and the data, calculates the entropy using the ID3 heuristic
	"""
	class_freq = {}
	data_entropy = 0.0

	# count the frequency of each class in the data set. 

	for row in data:
		#if we created an entry for the current class increment it 
		if(class_freq.has_key(row[targetAttr])):
			class_freq[row[targetAttr]] += 1.0
		else :
			# create an entry for the current class 
			class_freq[row[targetAttr]] = 1.0

	# calculate entropy of the data as sum over all classes

	for freq in class_freq.values():
		data_entropy += (-freq / len(data)) * math.log(freq / len(data), 2)

	return data_entropy


def gain(data, attribute, targetAttr):
	"""
	Calculates the gain of using the attribute to classify the data
	"""

	attribute_freq = {}
	subset_entropy = 0.0

	# count the frequency of the choosen (attribute) one

	for row in data:
		#print row
		#print attribute
		if(attribute_freq.has_key(row[attribute])):
			attribute_freq[row[attribute]] += 1.0
		else:
			attribute_freq[row[attribute]] = 1.0

	# calculate the subset_entropy

	for attr in attribute_freq.keys():
		attr_prob = attribute_freq[attr] / sum(attribute_freq.values())
		data_subset = [row for row in data if row[attribute] == attr]
		subset_entropy += attr_prob * entropy(data_subset, targetAttr)

	# gain is overall entropy minus subset_entropy

	return (entropy(data, targetAttr) - subset_entropy)


def create_decision_tree(examples, attributes, target_attribute, heuristic_funtion):
	"""
	Create the decision tree based on the ID3 heuristic passed in as the heuristic_funtion
	"""

	# make copies of data since python list are mutable
	examples = examples[:]
	# possible values for the attribute
	vals = [record[target_attribute] for record in examples]
	# If you reach a leaf node in the decision tree and have no examples left or 
	#the examples are equally split among multiple classes, then 
	#choose the class that is most frequent in the entire training set

	default = majority_value(examples, target_attribute)

	# if all the records in the dataset have the same (positive 1 or negative 0 classification)
	# then return that classification.
	if vals.count(vals[0]) == len(vals):
		return vals[0]
	# else if attributes are empty (reached leaf node) and there are still examples
	# choose the most frequent class among the instances of the leaf node
	elif not examples or (len(attributes) - 1) <= 0 :
		return default
	else:

		# choose the best attribute to classifiy data by using the ID3 heuristic gain funtion
		best = choose_attribute(examples, attributes, target_attribute, heuristic_funtion)

		# tree is represented as a dict of dicts. Instantiate the root with empty children
		tree = {best:{}}

		# for each possible values of best attribute create subtree/ child node
		for val in get_values(examples, best):
			subtree = create_decision_tree(
				get_examples(examples,best,val),
				[attr for attr in attributes if attr != best], # removing the chosen best attribute from the list here
				target_attribute,
				heuristic_funtion) 
			tree[best][val] = subtree
	return tree

########################## Helper Functions #####################
"""
The following are helper funtions used in creating the decision tree. These are
courtsey http://onlamp.com/pub/a/python/2006/02/09/ai_decision_trees.html
"""

"""
This module holds functions that are responsible for creating a new
decision tree and for using the tree for data classificiation.
"""

def majority_value(data, target_attr):
    """
    Creates a list of all values in the target attribute for each record
    in the data list object, and returns the value that appears in this list
    the most frequently.
    """
    data = data[:]
    return most_frequent([record[target_attr] for record in data])

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def unique(lst):
    """
    Returns a list made up of the unique values found in lst.  i.e., it
    removes the redundant values in lst.
    """
    lst = lst[:]
    unique_lst = []

    # Cycle through the list and add each value to the unique list only once.
    for item in lst:
        if unique_lst.count(item) <= 0:
            unique_lst.append(item)
            
    # Return the list with all redundant values removed.
    return unique_lst

def get_values(data, attr):
    """
    Creates a list of values in the chosen attribut for each record in data,
    prunes out all of the redundant values, and return the list.  
    """
    data = data[:]
    return unique([record[attr] for record in data])

def choose_attribute(data, attributes, target_attr, fitness):
    """
    Cycles through all the attributes and returns the attribute with the
    highest information gain (or lowest entropy).
    """
    data = data[:]
    best_gain = 0.0
    best_attr = None

    for attr in attributes:
        gain = fitness(data, attr, target_attr)
        if (gain >= best_gain and attr != target_attr):
            best_gain = gain
            best_attr = attr
                
    return best_attr

def get_examples(data, attr, value):
    """
    Returns a list of all the records in <data> with the value of <attr>
    matching the given value.
    """
    data = data[:]
    rtn_lst = []
    
    if not data:
        return rtn_lst
    else:
        record = data.pop()
        if record[attr] == value:
            rtn_lst.append(record)
            rtn_lst.extend(get_examples(data, attr, value))
            return rtn_lst
        else:
            rtn_lst.extend(get_examples(data, attr, value))
            return rtn_lst

def get_classification(record, tree):
    """
    This function recursively traverses the decision tree and returns a
    classification for the given record.
    """
    # If the current node is a string, then we've reached a leaf node and
    # we can return it as our answer
    if type(tree) == type("string"):
        return tree

    # Traverse the tree further until a leaf node is found.
    else:
        attr = tree.keys()[0]
        t = tree[attr][record[attr]]
        return get_classification(record, t)

def classify(tree, data):
    """
    Returns a list of classifications for each of the records in the data
    list as determined by the given decision tree.
    """
    data = data[:]
    classification = []
    
    for record in data:
        classification.append(get_classification(record, tree))

    return classification

