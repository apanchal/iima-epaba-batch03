"""
Project: Object Recognising Traffic Signs Using Deep Learning
Mentor: Prof. Ankur Sinha

This file contains all utility stuff.

@author: Ashish Panchal(aashish.panchal@gmail.com)
"""

import csv
class Util:

    class_lables = {}

    def __init__(self):
        # Empty constructor
        listtuples = []
        with open('../data/interim/signnames.csv', 'r') as csvfile:
            signnames = csv.reader(csvfile, delimiter=',')
            next(signnames, None)
            for row in signnames:
                listtuples.append((int(row[0]), row[1]))
            csvfile.close()
            class_lables = dict(listtuples)   

    

    def getsignLabel(classId):
        """
        A function that converts the class ids to the corresponding class name (name of traffic sign).
        Arguments:  classId - sign class id
        """
        return class_lables[classId]
