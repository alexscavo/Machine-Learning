import numpy
import scipy
import functions
import loadData

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



