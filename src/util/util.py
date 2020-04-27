"""
Project: Object Recognising Traffic Signs Using Deep Learning
Mentor: Prof. Ankur Sinha

This file contains all utility stuff.

@author: Ashish Panchal(aashish.panchal@gmail.com)
"""

import csv
class Util:

    def __init__(self):
        #print('Default called')
        # Empty constructor
        self.class_lables = {}
        listtuples = []
        with open('../data/interim/signnames.csv', 'r') as csvfile:
            signnames = csv.reader(csvfile, delimiter=',')
            next(signnames, None)
            for row in signnames:
                listtuples.append((int(row[0]), row[1]))
            csvfile.close()
            self.class_lables = dict(listtuples)

    

    def getsignLabel(self,classId):
        return self.class_lables[classId]
