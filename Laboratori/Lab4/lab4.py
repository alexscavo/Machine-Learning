import numpy
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg as scipy

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))


#---CALCOLO LOG PROBABILITY DENSITY FUNCTION---#
def logpdf_GAU_ND_Single_Sample(x, mu, C):
    
    M = x.shape[0]     # mu e' un array (M, 1)
    P = numpy.linalg.inv(C)     # inverse of the covariance matrix

    pdf = -(M/2)*numpy.log(2*numpy.pi) - 0.5 * numpy.linalg.slogdet(C)[1] - 0.5 * (x - mu).T @ P @ (x - mu)

    return pdf.ravel()

# Compute log-densities for N samples, arranged as a MxN matrix X (N stacked column vectors). The result is a 1-D array with N elements, corresponding to the N log-densities
def logpdf_GAU_ND_extended(X, mu, C):
    ll = [logpdf_GAU_ND_Single_Sample(X[:, i:i+1], mu, C) for i in range(X.shape[1])]
    return numpy.array(ll).ravel()


def loglikelihood(XND, mu_ML, C_ML):
    return logpdf_GAU_ND_extended(XND, mu_ML, C_ML).sum()



#------MAIN-----#
if __name__ == '__main__':

    plt.figure()
    XPlot = numpy.linspace(-8, 12, 1000)
    m = numpy.ones((1,1)) * 1.0
    C = numpy.ones((1,1)) * 2.0
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND_extended(vrow(XPlot), m, C)))
    plt.show()

    pdfGau = logpdf_GAU_ND_extended(vrow(XPlot), m, C)

    pdfSol = numpy.load('llGAU.npy')
    pdfGau = logpdf_GAU_ND_extended(vrow(XPlot), m, C)
    print(numpy.abs(pdfSol - pdfGau).max())
    print (numpy.abs(pdfSol - pdfGau).max())

    XND = numpy.load('XND.npy')
    mu = numpy.load('muND.npy')
    C = numpy.load('CND.npy')
    pdfSol = numpy.load('llND.npy')
    pdfGau = logpdf_GAU_ND_extended(XND, mu, C)
    print(numpy.abs(pdfSol - pdfGau).max())


    '''MAXIMUM LIKELIHOOD ESTIMATE'''
    # XND
    N = float(XND.shape[1])    # count the number of samples
    mu_ML = vcol(XND.mean(1))     #shape: (2, 1)
    XNDC = XND - mu_ML
    C_ML = (1/N) * (XNDC @ XNDC.T)  # covariance matrix

    ll = loglikelihood(XND, mu_ML, C_ML)
    print(mu_ML)
    print(C_ML)
    print(ll) 

    # X1D
    X1D = numpy.load('X1D.npy')
    N = float(X1D.shape[1])    # count the number of samples
    mu_ML = vcol(X1D.mean(1))     #shape: (2, 1)
    X1DC = X1D - mu_ML
    C_ML = (1/N) * (X1DC @ X1DC.T)  # covariance matrix

    print(mu_ML)
    print(C_ML)

    plt.figure()
    plt.hist(X1D.ravel(), bins = 50, density = True)
    XPlot = numpy.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND_extended(vrow(XPlot), mu_ML, C_ML)))
    plt.show()

    ll = loglikelihood(X1D, mu_ML, C_ML)
    print(ll)

