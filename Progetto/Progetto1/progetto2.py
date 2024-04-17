import numpy
import matplotlib
import matplotlib.pyplot as plt
import scipy
import plots
import functions
import loadData


def PCA_matrix(D, m):

    N = D.shape[1]      # total number of samples
    mu = D.mean(1)  # dataset mean (columns mean)

    DC = D - functions.mcol(mu)     #centered dataset

    C = float(1/N) * (DC @ DC.T)     # covariance matrix

    U, s, Vh = numpy.linalg.svd(C)      # svd on covariance matrix

    P = U[:, 0:m]

    return P


def LDA_matrix(D, L , m):

    Sb = 0
    Sw = 0

    mu_ds = functions.mcol(D.mean(1))   # dataset mean
    
    # Compute Sb and Sw
    for i in numpy.unique(L):   # for each label

        Di = D[:, L == i]       # consider only the samples with label = i
        DiC = Di - mu_ds        # center the data

        mu_class = functions.mcol(Di.mean(1))   # compute the class mean

        Sb += ((mu_class - mu_ds) @ (mu_class - mu_ds).T) * Di.shape[1]
        Sw += (Di @ Di.T) * (1/D.shape[1])

    # Solve the generalized eigenvalue problem

    s, U = scipy.linalg.eigh(Sb, Sw)    # compute the eigenvalues and eigenvectors

    W = U[:, ::-1][:, 0:m]  # consider only the first m eigenvectors (directions)

    return W





if __name__ == '__main__':

    # load the data
    D, L = loadData.load('trainData.txt')

    print(L)

    #(DTR, LTR), (DVAL, LVAL)  = functions.split_training_test_dataset(D, L)    serve per la classificazione che verra' fatta dopo!


    #
    #-----PCA-----
    #
    P_pca = PCA_matrix(D, 6)       # PCA matrix containing eigenvectors in descending order

    DP_pca = P_pca.T @ D    # project data on the new, using P.T since we want descending ordered eigenvectors

    print("P_pca = \n", P_pca)

    plots.plot_histograms("plots_p2/PCA", DP_pca, L, range(6))


    #
    #-----LDA-----
    #
    W_lda = LDA_matrix(D, L, 1)     # only 1 dimension since we have just 2 classes and LDA finds C-1 directions

    print("W_lda = \n", W_lda)

    DP_lda = W_lda.T @ D        # project the dataset applying LDA matrix

    plots.plot_histograms("plots_p2/LDA", DP_lda, L, range(1))