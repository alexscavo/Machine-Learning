import sklearn.datasets
import numpy
import scipy

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def compute_mean_covariance(D):
    N = float(D.shape[1])
    mu = mcol(D.mean(1))
    DC = D - mu
    C = (DC @ DC.T) / N

    return mu, C



# mi dice quanto e' probabile che ciascun sample di x appartenga alla classe di cui i parametri sono mu e C
def logpdf_GAU_ND(x, mu, C):
    
    M = x.shape[0]     # mu e' un array (M, 1)
    P = numpy.linalg.inv(C)     # inverse of the covariance matrix

    pdf = -(M/2)*numpy.log(2*numpy.pi) - 0.5 * numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

    return pdf


def compute_parameters_MVG(D, L):
    labels = set(L)     # ottengo le singole label
    parameters = {}     # creo il dict che conterra' i parametri
    for label in labels:    
        DX = D[:, L == label]   # considero solo i sample di classe label
        parameters[label] = compute_mean_covariance(DX)     # inserisco mu e C nel dict in posizione label

    return parameters


def class_conditional_prob(D, parameters):
    S = numpy.zeros((len(parameters), D.shape[1]))      # 4x50

    for label in range(S.shape[0]):     # per ogni classe
        S[label, :] = logpdf_GAU_ND(D, parameters[label][0], parameters[label][1])  # calcolo la class conditional probability di ciascun sample data la classe in questione

    return S

def compute_parameters_naive_bayes(D, L):
    labels = set(L)
    parameters = {}
    for label in labels:
        DX = D[:, L== label]
        mu, C = compute_mean_covariance(DX)
        C = C * numpy.identity(C.shape[0])  # tolgo tutti gli elementi che non sono sulla diagonale, moltiplicando la matrice delle cov con la amtrice identita'
        parameters[label] = (mu, C)

    return parameters

def compute_parameters_tied(D, L):
    labels = set(L)
    parameters = {}
    C_global = 0
    for label in labels:
        DX = D[:, L == label]
        mu, C = compute_mean_covariance(DX)
        parameters[label] = (mu, C * DX.shape[1])

    SW = (parameters[0][1] + parameters[1][1] + parameters[2][1]) * (1/D.shape[1])

    parameters[0] = (parameters[0][0], SW) 
    parameters[1] = (parameters[1][0], SW)
    parameters[2] = (parameters[2][0], SW)

    return parameters



if __name__ == '__main__':

    D, L = load_iris()
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # ----- MULTIVARIATE GAUSSIAN CLASSIFIER -----
    parameters = compute_parameters_MVG(DTR, LTR)   # calcolo i parametri per il train set

    # 1 - COMPUTE THE LIKELIHOOD
    S = class_conditional_prob(DVAL, parameters)

    #print(S)
    # 2 - COMPUTE THE POSTERIOR PROBABILITY
    prior_prob = mcol(numpy.ones(3)/3.)   # ottengo un vettore colonna di 3 elementi contenenti 1/3, che indica la prior probability di ciascuna classe

    SJoint_prob = S + numpy.log(prior_prob)
    #print('---------my joint density: ------------\n', SJoint_prob, '\n')

    marginal_densities = mrow(scipy.special.logsumexp(SJoint_prob, axis=0))

    SPost = SJoint_prob - marginal_densities   # faccio la sottrazione dato che ho da calcolare il logaritmo di una divisione e ho gia' calcolato il logaritmo degli operandi


    #print('\n-----POSTERIOR PROBABILITY-----\n', SPost)

    predicted_val = SPost.argmax(0)
    print('\n\nMVG - predicted value: ', predicted_val)
    print('MVG - Errors: \n', predicted_val-LVAL)
    print('MVG - Error rate: ', (predicted_val-LVAL).sum()/float(LVAL.size)*100, '%')

    # ----- NAIVE BAYES -----
    parameters_naive = compute_parameters_naive_bayes(DTR, LTR)

    S = class_conditional_prob(DVAL, parameters_naive)
    prior_prob = mcol(numpy.ones(3)/3.)  
    SJoint_prob = S + numpy.log(prior_prob)
    marginal_densities = mrow(scipy.special.logsumexp(SJoint_prob, axis=0))
    SPost = SJoint_prob - marginal_densities   
    predicted_val = SPost.argmax(0)
    print('\n\nNaive Bayes - predicted value: ', predicted_val)
    print('Naive Bayes - Errors: \n', predicted_val-LVAL)
    print('Naive Bayes - Error rate: ', (predicted_val-LVAL).sum()/float(LVAL.size)*100, '%')

    # ----- TIED GAUSSIAN -----
    parameters_tied = compute_parameters_tied(DTR, LTR)

    S = class_conditional_prob(DVAL, parameters_tied)
    prior_prob = mcol(numpy.ones(3)/3.)  
    SJoint_prob = S + numpy.log(prior_prob)
    marginal_densities = mrow(scipy.special.logsumexp(SJoint_prob, axis=0))
    SPost = SJoint_prob - marginal_densities   
    predicted_val = SPost.argmax(0)
    print('\n\nTied Gaussian - predicted value: ', predicted_val)
    print('Tied Gaussian - Errors: \n', predicted_val-LVAL)
    print('Tied Gaussian - Error rate: ', (predicted_val-LVAL).sum()/float(LVAL.size)*100, '%')