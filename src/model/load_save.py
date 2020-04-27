"""
Project: Object Recognising Traffic Signs Using Deep Learning
Mentor: Prof. Ankur Sinha

This file contains stuff to save and load the datastructures we use (especially the model) and scalar values for later evaluation.

@author: Ashish Panchal(aashish.panchal@gmail.com)
"""

import  pickle as p



class LoadSave:
    

    def __init__(self):
        self.model_base_path = "../models/"
        self.suffix = ".model"

    def saveclassifier(self, model, filename):
        """
        Save all data into one .model file
        Arguments:  
        model - classifier object
        filename - filename without .model
        """
        p.dump(model, open(self.model_base_path+filename+self.suffix, 'wb'))

    def loadclassifier(self, filename):
         """
         Read all data from one .model file created with LoadSave.saveclassifier
         Arguments:  filename - filename without .model
         """
         clf = p.load(open(self.model_base_path+filename+self.suffix, 'rb'))
         return clf