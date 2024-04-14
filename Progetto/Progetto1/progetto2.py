import numpy
import matplotlib
import matplotlib.pyplot as plt
import plots
import functions
import loadData


def PCA_matrix(D, m):

    N = D.shape[1]      # total number of samples
    mu = D.mean(1)  # dataset mean (columns mean)

    DC = D - functions.mcol(mu)     #centered dataset

    C = (1/N) * (DC @ DC.T)     # covariance matrix

    #s, U = numpy.linalg.eigh(C)     # s = sigma matrix, containing eigenvalues in ascending order
                                    # U = eigenvectors ordered accordingly
    #P = U[:, ::-1][:, 0:m]      # retrieve first m eigenvectors

    U, s, Vh = numpy.linalg.svd(C)      # svd on covariance matrix

    P = U[:, 0:m]

    return P



if __name__ == '__main__':

    # load the data
    D, L = loadData.load('trainData.txt')

    (DTR, LTR), (DVAL, LVAL)  = functions.split_training_test_dataset(D, L)


    #
    #-----PCA-----
    #
    P_pca = PCA_matrix(DTR, 6)       # PCA matrix containing eigenvectors in descending order

    DTR_P = P_pca.T @ DTR    # project data on the new, using P.T since we want descending ordered eigenvectors

    print("P_pca = \n", P_pca)

    plots.plot_histograms("plots_p2", DTR_P, LTR, [0, 1, 2, 3, 4, 5])