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

    #(DTR, LTR), (DVAL, LVAL)  = functions.split_training_test_dataset(D, L)    serve per la classificazione che verra' fatta dopo!


    #
    #-----PCA-----
    #
    P_pca = PCA_matrix(D, 6)       # PCA matrix containing eigenvectors in descending order

    DTR_P = P_pca.T @ D    # project data on the new, using P.T since we want descending ordered eigenvectors

    print("P_pca = \n", P_pca)

    plots.plot_histograms("plots_p2", DTR_P, L, [0, 1, 2, 3, 4, 5])


    #
    #-----LDA-----
    #
    W_lda = LDA_matrix(D, L, 1)     # only 1 dimension since we have just 2 classes and LDA finds C-1 directions

    print("W_lda = \n", W_lda)
    #plots.plot_histograms()