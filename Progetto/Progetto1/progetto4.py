import numpy
import scipy
import functions
import loadData
import plots

# it tells me the likelihood that a sample x has class whose parameters are mu and C
def logpdf_GAU_ND(x, mu, C):
    
    M = x.shape[0]     # mu e' un array (M, 1)
    P = numpy.linalg.inv(C)     # inverse of the covariance matrix

    pdf = -(M/2)*numpy.log(2*numpy.pi) - 0.5 * numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

    return pdf

# it computes the mean and covariance, given D and L
def compute_parameters_MVG(D, L):
    labels = set(L)     # ottengo le singole label
    parameters = {}     # creo il dict che conterra' i parametri
    for label in labels:    
        DX = D[:, L == label]   # considero solo i sample di classe label
        parameters[label] = functions.compute_mean_covariance(DX)     # inserisco mu e C nel dict in posizione label

    return parameters

def compute_parameters_tied(D, L):
    labels = set(L)
    parameters = {}
    means = {}
    C_global = 0
    for label in labels:
        DX = D[:, L == label]
        mu, C = functions.compute_mean_covariance(DX)
        C_global += C * DX.shape[1]
        means[label] = mu

    SW = C_global / D.shape[1]

    for label in labels:
        parameters[label] = (means[label], SW)

    return parameters

def compute_parameters_naive_bayes(D, L):
    labels = set(L)
    parameters = {}
    for label in labels:
        DX = D[:, L== label]
        mu, C = functions.compute_mean_covariance(DX)
        C = C * numpy.identity(C.shape[0])  # remove every off-diagonal element multiplying the covariance matrix with the identity matrix
        parameters[label] = (mu, C)

    return parameters

def compute_llr(D, parameters):
    pdf_1 = logpdf_GAU_ND(D, parameters[1][0], parameters[1][1])
    pdf_0 = logpdf_GAU_ND(D, parameters[0][0], parameters[0][1])

    return pdf_1 - pdf_0 

def compute_error_rate(P, L):
    return ((P != L).sum()/float(L.size))*100

def compute_predictions(D, class_prior_prob, llr, threshold):
    PVAL = numpy.zeros(DVAL.shape[1], dtype = numpy.int32)
    class_prior_prob = [0.5, 0.5]   # class prior probabilities

    threshold = -numpy.log(class_prior_prob[1]/class_prior_prob[0])     # compute the threshold as -log( P(C == 1)/P(C == 0) )

    PVAL[llr >= threshold] = 1
    PVAL[llr < threshold] = 0

    return PVAL

def get_covariance_per_class(parameters, L):

    labels = set(L)
    C_per_class = {}

    for label in labels:
        C_per_class[label] = parameters[label][1]

    return C_per_class

if __name__ == '__main__':

    D, L = loadData.load('trainData.txt')      # get the data and labels from the dataset

    class_prior_prob = [0.5, 0.5]   # class prior probabilities
    threshold = -numpy.log(class_prior_prob[1]/class_prior_prob[0])     # compute the threshold as -log( P(C == 1)/P(C == 0) )

    (DTR, LTR), (DVAL, LVAL) = functions.split_training_test_dataset(D, L)  # obtain training and validation data

    # ----- MVG -----
    parameters = compute_parameters_MVG(DTR, LTR)   # compute training parameters with MVG model
    llr = compute_llr(DVAL, parameters)
    #print(llr)

    # predictions:
    PVAL = compute_predictions(DVAL, class_prior_prob, llr, threshold)
    print('MVG model -  classification error rate (threshold: ',threshold, '): ', compute_error_rate(PVAL, LVAL), '%')

    # ----- TIED GAUSSIAN -----
    parameters = compute_parameters_tied(DTR, LTR)  # compute training parameters with tied Gaussian model
    llr = compute_llr(DVAL, parameters)
    #print(llr)

    # predictions:
    PVAL = compute_predictions(DVAL, class_prior_prob, llr, threshold)
    print('Tied Gaussian model -  classification error rate (threshold: ',threshold, '): ', compute_error_rate(PVAL, LVAL), '%')

    # ----- NAIVE BAYES GAUSSIAN -----
    parameters = compute_parameters_naive_bayes(DTR, LTR)
    llr = compute_llr(DVAL, parameters)
    #print(llr)


    # predictions
    PVAL = compute_predictions(DVAL, class_prior_prob, llr, threshold)
    print('Naive Bayes Gaussian model -  classification error rate (threshold: ',threshold, '): ', compute_error_rate(PVAL, LVAL), '%')

    # ----- COVARIANCE MATRIX -----
    parameters = compute_parameters_MVG(DTR, LTR)   # we consider the MVG model parameters
    C_per_class = get_covariance_per_class(parameters, LTR)

    print('\n--------COVARIANCE MATRICES--------\n')
    for label in set(LTR):

        C = C_per_class[label]

        print('Covariance matrix - class', label, ':')
        
        functions.print_matrix(numpy.round(C, 2))

        Corr = C / (functions.mcol(C.diagonal()**0.5) * functions.mrow(C.diagonal()**0.5))

        print('\nCorrelation matrix - class', label, ':')
        functions.print_matrix(numpy.round(Corr, 4))
        
        print()


    # prova plot naive bayes - forse da rimuovere
    '''parameters = compute_parameters_MVG(DTR, LTR)
    DVAL0 = DVAL[:, LVAL == 0]
    DVAL1 = DVAL[:, LVAL == 1]

    for i in range(6):    
        plots.plots_pdf_GAU("plots_p4", i, 0, functions.mrow(numpy.sort(DVAL0[i, :])), parameters[0][0], parameters[0][1])

    for i in range(6):    
        plots.plots_pdf_GAU("plots_p4", i, 1, functions.mrow(numpy.sort(DVAL1[i, :])), parameters[1][0], parameters[1][1])'''

    
    #
    # ----- PROVE SENZA FEATURES 5 E 6 -----
    #

    DTR = DTR[0:5, :]
    DVAL = DVAL[0:5, :]

    # ----- MVG -----
    parameters = compute_parameters_MVG(DTR, LTR)   # compute training parameters with MVG model
    llr = compute_llr(DVAL, parameters)
    #print(llr)

    # predictions:
    PVAL = compute_predictions(DVAL, class_prior_prob, llr, threshold)
    print('MVG model optimized -  classification error rate (threshold: ',threshold, '): ', compute_error_rate(PVAL, LVAL), '%')

    # ----- TIED GAUSSIAN -----
    parameters = compute_parameters_tied(DTR, LTR)  # compute training parameters with tied Gaussian model
    llr = compute_llr(DVAL, parameters)
    #print(llr)

    # predictions:
    PVAL = compute_predictions(DVAL, class_prior_prob, llr, threshold)
    print('Tied Gaussian model optimized -  classification error rate (threshold: ',threshold, '): ', compute_error_rate(PVAL, LVAL), '%')

    # ----- NAIVE BAYES GAUSSIAN -----
    parameters = compute_parameters_naive_bayes(DTR, LTR)
    llr = compute_llr(DVAL, parameters)
    #print(llr)


    # predictions
    PVAL = compute_predictions(DVAL, class_prior_prob, llr, threshold)
    print('Naive Bayes Gaussian model optimized -  classification error rate (threshold: ',threshold, '): ', compute_error_rate(PVAL, LVAL), '%')

    # ----- COVARIANCE MATRIX -----
    parameters = compute_parameters_MVG(DTR, LTR)   # we consider the MVG model parameters
    C_per_class = get_covariance_per_class(parameters, LTR)

    print('\n--------COVARIANCE MATRICES--------\n')
    for label in set(LTR):

        C = C_per_class[label]

        print('Covariance matrix optimized - class', label, ':')
        
        functions.print_matrix(numpy.round(C, 2))

        Corr = C / (functions.mcol(C.diagonal()**0.5) * functions.mrow(C.diagonal()**0.5))

        print('\nCorrelation matrix optimized - class', label, ':')
        functions.print_matrix(numpy.round(Corr, 4))
        
        print()

    #
    # ----- JUST FEATURES 1 AND 2 CLASSIFICATIONS ----- 
    #
    DTR = DTR[0:2, :]
    DVAL = DVAL[0:2, :]

    # ----- MVG -----
    parameters = compute_parameters_MVG(DTR, LTR)   # compute training parameters with MVG model
    llr = compute_llr(DVAL, parameters)
    #print(llr)

    # predictions:
    PVAL = compute_predictions(DVAL, class_prior_prob, llr, threshold)
    print('MVG model features 1-2 -  classification error rate (threshold: ',threshold, '): ', compute_error_rate(PVAL, LVAL), '%')

    # ----- TIED GAUSSIAN -----
    parameters = compute_parameters_tied(DTR, LTR)  # compute training parameters with tied Gaussian model
    llr = compute_llr(DVAL, parameters)

    # predictions:
    PVAL = compute_predictions(DVAL, class_prior_prob, llr, threshold)
    print('Tied Gaussian model features 1-2 -  classification error rate (threshold: ',threshold, '): ', compute_error_rate(PVAL, LVAL), '%')

    # ----- JUST FEATURES 3 AND 4 CLASSIFICATIONS ----- 
    DTR = DTR[2:4, :]
    DVAL = DVAL[2:4, :]

    # ----- MVG -----
    parameters = compute_parameters_MVG(DTR, LTR)   # compute training parameters with MVG model
    llr = compute_llr(DVAL, parameters)
    #print(llr)

    # predictions:
    PVAL = compute_predictions(DVAL, class_prior_prob, llr, threshold)
    print('MVG model features 3-4 -  classification error rate (threshold: ',threshold, '): ', compute_error_rate(PVAL, LVAL), '%')

    # ----- TIED GAUSSIAN -----
    parameters = compute_parameters_tied(DTR, LTR)  # compute training parameters with tied Gaussian model
    llr = compute_llr(DVAL, parameters)

    # predictions:
    PVAL = compute_predictions(DVAL, class_prior_prob, llr, threshold)
    print('Tied Gaussian model features 3-4 -  classification error rate (threshold: ',threshold, '): ', compute_error_rate(PVAL, LVAL), '%')

    #
    # ----- PREPROCESSING WITH PCA -----
    #
    print('\n----- Preprocessing with PCA -----\n')
    (DTR, LTR), (DVAL, LVAL) = functions.split_training_test_dataset(D, L)

    for m in range(2, 7):
        print('m =', m)
        DTR_pca = functions.PCA_matrix(DTR, m) 

        DTRP_pca = DTR_pca.T @ DTR    # project the data over the new subspace
        DVALP_pca = DTR_pca.T @ DVAL

        # ----- MVG -----
        parameters = compute_parameters_MVG(DTRP_pca, LTR)   # compute training parameters with MVG model
        llr = compute_llr(DVALP_pca, parameters)

        # predictions:
        PVAL = compute_predictions(DVAL, class_prior_prob, llr, threshold)
        print('PCA + MVG model -  classification error rate (threshold: ',threshold, '): ', compute_error_rate(PVAL, LVAL), '%')

        # ----- TIED GAUSSIAN -----
        parameters = compute_parameters_tied(DTRP_pca, LTR)  # compute training parameters with tied Gaussian model
        llr = compute_llr(DVALP_pca, parameters)

        # predictions:
        PVAL = compute_predictions(DVALP_pca, class_prior_prob, llr, threshold)
        print('PCA + Tied Gaussian model -  classification error rate (threshold: ',threshold, '): ', compute_error_rate(PVAL, LVAL), '%')

        # ----- NAIVE BAYES GAUSSIAN -----
        parameters = compute_parameters_naive_bayes(DTRP_pca, LTR)
        llr = compute_llr(DVALP_pca, parameters)
        #print(llr)


        # predictions
        PVAL = compute_predictions(DVALP_pca, class_prior_prob, llr, threshold)
        print('Naive Bayes Gaussian model -  classification error rate (threshold: ',threshold, '): ', compute_error_rate(PVAL, LVAL), '%')





