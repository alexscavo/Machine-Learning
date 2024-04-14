import numpy
import matplotlib
import matplotlib.pyplot as plt


def mcol(v):    #transposed vector vertically
    return v.reshape(v.size, 1)

def mrow(v):    #transposed vector horitzonally
    return v.reshape(1, v.size)


def load(fileName):     #function to load the dataset
    
    DList = []      #list of features
    LabelList = []  #list of labels (already numbers so no need to translate them with a dict)

    with open(fileName) as f:
        
        for line in f:
            try:
                features = line.split(',')[0:-1]    #select only the first 6 elements
                label = line.split(',')[-1]          #select the last element
        
                features = mcol(numpy.array([float(i) for i in features]))   #convert each element into a float value, obtain the array and transpose it

                DList.append(features)      #add the features array to the list of attributes
                LabelList.append(label)     #add the label to the list of labels
            except:
                pass
        
    return numpy.hstack(DList), numpy.array(LabelList, dtype = numpy.int32)


def plot_histograms(D, L, featuresOfInterest):
    
    D0 = D[:, L == 0]   #separate the samples based on the label value
    D1 = D[:, L == 1]

    for feat in featuresOfInterest:
        plt.figure()
        plt.xlabel('Feature %d' % (feat))
        plt.hist(D0[feat, :], bins = 10, density = True, alpha = 0.4, label = 'Counterfeit')
        plt.hist(D1[feat, :], bins = 10, density = True, alpha = 0.4, label = 'Genuine')

        plt.legend()
        plt.tight_layout()
        plt.savefig('plots_p1/hist_%d.pdf' % (feat))
    plt.show()




def plot_scatter(D, L, featureOfInterest):
    
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    for feat1 in featureOfInterest:
        for feat2 in featureOfInterest:

            if feat1 == feat2 or feat1 > feat2:
                continue
            
            plt.figure()
            plt.xlabel('Feature %d' % (feat1))
            plt.ylabel('Feature %d' % (feat2))
            plt.scatter(D0[feat1, :], D0[feat2, :], label = 'Counterfeit')
            plt.scatter(D1[feat1, :], D1[feat2, :], label = 'Genuine')

            plt.legend()
            plt.tight_layout()
            plt.savefig('plots_p1/scatter_%d_%d.pdf' % (feat1, feat2))
        plt.show()


if __name__ == '__main__':

    #change default font size - comment to use default values
    plt.rc('font', size = 16)
    plt.rc('xtick', labelsize = 16)
    plt.rc('ytick', labelsize = 16)


    D, L = load('trainData.txt')
    
     
    #feature 1 - feature 2
    plot_histograms(D, L, [0, 1])
    plot_scatter(D, L, [0, 1])

    #feature 3 - feature 4
    plot_histograms(D, L, [2, 3])
    plot_scatter(D, L, [2, 3])

    #feature 5 - feature 6
    plot_histograms(D, L, [4, 5])
    plot_scatter(D, L, [4, 5])
    
    '''
    mu = D.mean(1).reshape((D.shape[0], 1))
    var = D.var(1)
    print('Mean =\n {} \n\n Variance = {}\n'.format(mu, var))
    '''

    for cls in [0, 1]:
        print('class', cls)
        Dcls = D[:, L == cls]
        mu = Dcls.mean(1).reshape(Dcls.shape[0], 1)
        print('Mean:')
        print(mu)
        
        var = Dcls.var(1)
        print('Variance', var)
        print()
    