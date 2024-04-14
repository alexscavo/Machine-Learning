import numpy
import functions


def load(fileName):     #function to load the dataset
    
    DList = []      #list of features
    LabelList = []  #list of labels (already numbers so no need to translate them with a dict)

    with open(fileName) as f:
        
        for line in f:
            try:
                features = line.split(',')[0:-1]    #select only the first 6 elements
                label = line.split(',')[-1]          #select the last element
        
                features = functions.mcol(numpy.array([float(i) for i in features]))   #convert each element into a float value, obtain the array and transpose it

                DList.append(features)      #add the features array to the list of attributes
                LabelList.append(label)     #add the label to the list of labels
            except:
                pass
        
    return numpy.hstack(DList), numpy.array(LabelList, dtype = numpy.int32)