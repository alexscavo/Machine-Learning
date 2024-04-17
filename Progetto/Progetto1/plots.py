import numpy
import matplotlib
import matplotlib.pyplot as plt


def plot_histograms(folder, D, L, featuresOfInterest):
    
    D0 = D[:, L == 0]   #separate the samples based on the label value
    D1 = D[:, L == 1]

    for feat in featuresOfInterest:
        plt.figure()
        plt.xlabel('Direction %d' % (feat))
        plt.hist(D0[feat, :], bins = 40, density = True, alpha = 0.4, label = 'Counterfeit')
        plt.hist(D1[feat, :], bins = 40, density = True, alpha = 0.4, label = 'Genuine')

        plt.legend()
        plt.tight_layout()
        path = f"{folder}/hist_{feat}.pdf"
        plt.savefig(path)
    plt.show()




def plot_scatter(folder, D, L, featureOfInterest):
    
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
            path = f"{folder}/scatter_{feat1}_{feat2}.pdf"
            plt.savefig(path)
        plt.show()