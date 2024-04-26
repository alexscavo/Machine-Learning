import numpy
import matplotlib.pyplot as plt
import plots
import functions
import loadData


def logpdf_GAU_ND_singleSample(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu).T @ P @ (x-mu)).ravel()

def logpdf_GAU_ND_extended(X, mu, C):

    ll = [logpdf_GAU_ND_singleSample(X[:, i:i+1], mu, C) for i in range(X.shape[1])]   # log-likelihood

    return numpy.array(ll).ravel()  # consider the result as a 1D array with N elements, corresponfing to the N log densities


def compute_Mean_Covariance(D):

    N = float(D.shape[0])
    mu = functions.mcol(D.mean(1))
    DC = D - mu
    C = (1/N) * (DC @ DC.T)

    return mu, C

def loglikelihood(X, mu_ML, C_ML):
    return logpdf_GAU_ND_extended(X, mu_ML, C_ML).sum()


if __name__ == '__main__':

    # load the data
    D, L = loadData.load('trainData.txt')

    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    for i in range(6):
        
        X1D = functions.mrow(numpy.sort(D0[i, :]))
        print(X1D.shape)
        mu_ML, C_ML = compute_Mean_Covariance(X1D)  # compute mean and covariance for each class 
        print("\nD0 = \n", X1D)
        print("\nmu = ", mu_ML)
        #ll = loglikelihood(X1D, mu_ML, C_ML)
        #print("\n ll class 0 feature ", i, " = \n")
        #print(ll)

        plt.figure()
        plt.plot(X1D.ravel(), numpy.exp(logpdf_GAU_ND_extended(X1D, mu_ML, C_ML)))
        plt.hist(X1D.ravel(), bins = 50, density = True)
        plt.show()

