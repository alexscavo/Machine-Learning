import matplotlib.pyplot
import numpy
import matplotlib
import matplotlib.pyplot as plt
import progetto3


def plot_histograms(folder, D, L, featuresOfInterest):
    
    D0 = D[:, L == 0]   #separate the samples based on the label value
    D1 = D[:, L == 1]

    for feat in featuresOfInterest:
        plt.figure()
        plt.xlabel('Feature %d' % (feat+1))
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

    feat1 = featureOfInterest[0]
    feat2 = featureOfInterest[1]

    '''for feat1 in featureOfInterest:
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
        plt.show()'''
    plt.figure()
    plt.xlabel('Feature %d' % (feat1+1))
    plt.ylabel('Feature %d' % (feat2+1))
    plt.scatter(D0[feat1, :], D0[feat2, :], label = 'Counterfeit')
    plt.scatter(D1[feat1, :], D1[feat2, :], label = 'Genuine')

    plt.legend()
    plt.tight_layout()
    path = f"{folder}/scatter_{feat1}_{feat2}.pdf"
    plt.savefig(path)
    plt.show()


def plots_pdf_GAU(folder, i, c, X1D, mu_ML, C_ML):
    plt.figure()
    plt.plot(X1D.ravel(), numpy.exp(progetto3.logpdf_GAU_ND_extended(X1D, mu_ML, C_ML)))
    plt.hist(X1D.ravel(), bins = 50, density = True)
    path = f"{folder}/plot_{i}_class{c}.pdf"
    plt.savefig(path)
    plt.show()

def plot_lab8(min_DCF, act_DCF, _lambda):
    plt.figure(figsize=(10, 6))
    
    plt.xscale('log', base=10)
    plt.plot(_lambda, min_DCF, label='Minimum DCF', marker='o')
    plt.plot(_lambda, act_DCF, label='Actual DCF', marker='x')
    
    plt.xlabel('λ (Regularization strength)')
    plt.ylabel('DCF')
    plt.title('DCF vs λ for Logistic Regression')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_lab9(title, min_DCF, act_DCF, _lambda):
    plt.figure(figsize=(10, 6))
    
    plt.xscale('log', base=10)
    plt.plot(_lambda, min_DCF, label='Minimum DCF', marker='o')
    plt.plot(_lambda, act_DCF, label='Actual DCF', marker='x')
    
    plt.xlabel('C (Regularization strength)')
    plt.ylabel('DCF')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_gmm(min_DCF, act_DCF, num_components):
    plt.figure(figsize=(10, 6))
    
    plt.plot(num_components, min_DCF, label='Minimum DCF', marker='o')
    plt.plot(num_components, act_DCF, label='Actual DCF', marker='x')
    
    plt.xlabel('Number of Components (True Class)')
    plt.ylabel('DCF')
    plt.title('DCF vs Number of Components for GMM')
    plt.legend()
    plt.grid(True)
    plt.show()

