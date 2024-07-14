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

        mu_class = functions.mcol(Di.mean(1))   # compute the class mean
        DiC = Di - mu_class        # center the data

        Sb += ((mu_class - mu_ds) @ (mu_class - mu_ds).T) * DiC.shape[1]
        Sw += (DiC @ DiC.T) 

    Sw = Sw / D.shape[1]
    Sb = Sb / D.shape[1]

    # Solve the generalized eigenvalue problem

    s, U = scipy.linalg.eigh(Sb, Sw)    # compute the eigenvalues and eigenvectors

    W = U[:, ::-1][:, 0:m]  # consider only the first m eigenvectors (directions)

    return W

def find_best_treshold(treshold_base, DVALP, LVAL):

    treshold = treshold_base
    min = 100

    for i in range(120*10**5):
        treshold += 10**-5
        PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
        PVAL[DVALP[0] >= treshold] = 1
        PVAL[DVALP[0] < treshold] = 0

        perc = (PVAL != LVAL).sum() / float(LVAL.size) *100

        if perc < min:
            min = perc 
            best_PVAL = PVAL
            best_treshold = treshold
            best_perc = perc

        treshold -= 2*10**-5
        PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
        PVAL[DVALP[0] >= treshold] = 1
        PVAL[DVALP[0] < treshold] = 0

        perc = (PVAL != LVAL).sum() / float(LVAL.size) *100

        if perc < min:
            min = perc 
            best_PVAL = PVAL
            best_treshold = treshold
            best_perc = perc
        

    return best_treshold, best_PVAL, best_perc


if __name__ == '__main__':

    # load the data
    D, L = loadData.load('trainData.txt')

    #
    #-----PCA-----
    #
    P_pca = PCA_matrix(D, 6)       # PCA matrix containing eigenvectors in descending order

    DP_pca = P_pca.T @ D    # project data on the new, using P.T since we want descending ordered eigenvectors

    print("P_pca = \n", P_pca)

    #plots.plot_histograms("plots_p2/PCA", DP_pca, L, range(6))

    #plots.plot_scatter("plots_p2/PCA", DP_pca, L, [0, 1])
    #plots.plot_scatter("plots_p2/PCA", DP_pca, L, [2, 3])
    #plots.plot_scatter("plots_p2/PCA", DP_pca, L, [4, 5])


    #
    #-----LDA-----
    #
    W_lda = LDA_matrix(D, L, 1)     # only 1 dimension since we have just 2 classes and LDA finds C-1 directions

    print("W_lda = \n", W_lda)

    DP_lda = W_lda.T @ D        # project the dataset applying LDA matrix

    #plots.plot_histograms("plots_p2/LDA", DP_lda, L, range(1))


    #
    #-----LDA for classification
    #
    print('-'*40)
    print('LDA for classification')
    (DTR, LTR), (DVAL, LVAL)  = functions.split_training_test_dataset(D, L)     # split the dataset into training data and validation data. Same thing for the labels

    DTR_lda = LDA_matrix(DTR, LTR, 1)   # compute the LDA matrix over training data

    DTRP_lda = DTR_lda.T @ DTR      # project dataset samples over the direction found by LDA

    # we're interested in the mean of class true (L==1) being larger than the mean of class false (L==0)
    if DTRP_lda[0, LTR == 1].mean() < DTRP_lda[0, LTR == 0].mean():     
        DTR_lda = -DTR_lda
        DTRP_lda = DTR_lda.T @ DTR

    DVALP_lda = DTR_lda.T @ DVAL     # project validation dataset over trained model

    treshold = (DTRP_lda[0, LTR == 0].mean() + DTRP_lda[0, LTR == 1].mean()) / 2.0

    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[DVALP_lda[0] >= treshold] = 1
    PVAL[DVALP_lda[0] < treshold] = 0

    print('Treshold: ', treshold)
    print('Number of errors:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.3f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))


    treshold = -0.10504437678623708

    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[DVALP_lda[0] >= treshold] = 1
    PVAL[DVALP_lda[0] < treshold] = 0

    print('Treshold: ', treshold)
    print('Number of errors:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.3f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))

    treshold = 0.2

    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[DVALP_lda[0] >= treshold] = 1
    PVAL[DVALP_lda[0] < treshold] = 0

    print('Treshold: ', treshold)
    print('Number of errors:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.3f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))

   
   #finds the best threshold
    '''treshold = (DTRP_lda[0, LTR == 0].mean() + DTRP_lda[0, LTR == 1].mean()) / 2.0
    best_treshold, best_PVAL, best_perc = find_best_treshold(treshold, DVALP_lda, LVAL)
    print('Best Treshold: ', best_treshold)
    print('Number of errors:', (best_PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.3f%%' % ( (best_PVAL != LVAL).sum() / float(LVAL.size) *100 ))'''


    #
    #-----PCA + LDA-----
    #
    print('-'*40)
    print('PCA + LDA')
    for m in range(2, 7):
        DTR_pca = PCA_matrix(DTR, m)

        DTRP_pca = DTR_pca.T @ DTR    # now I have the reduced dimensionality applied on the training dataset
        DVALP_pca = DTR_pca.T @ DVAL  # now I have the reduced dimensionality applied on the validation dataset

        DTR_lda = LDA_matrix(DTRP_pca, LTR, 1)     
        DTRP_lda = DTR_lda.T @ DTRP_pca   # now I apply LDA on projected training set with PCA
        # we're interested in the mean of class true (L==1) being larger than the mean of class false (L==0)
        if DTRP_lda[0, LTR == 1].mean() < DTRP_lda[0, LTR == 0].mean():     
            DTR_lda = -DTR_lda
            DTRP_lda = DTR_lda.T @ DTRP_pca

        DVALP_lda = DTR_lda.T @ DVALP_pca

        treshold = (DTRP_lda[0, LTR == 0].mean() + DTRP_lda[0, LTR == 1].mean()) / 2.0
        
        PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
        PVAL[DVALP_lda[0] >= treshold] = 1
        PVAL[DVALP_lda[0] < treshold] = 0

        print('\nm = ', m)
        print('Treshold: ', treshold)
        print('Number of errors:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
        print('Error rate: %.3f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))
        print()

        #plots.plot_histograms("plots_p2/LDA", DVALP_lda, LVAL, range(1))
