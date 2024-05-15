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
    C = (DC @ D.T) / N

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


if __name__ == '__main__':

    D, L = load_iris()
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # ----- MULTIVARIATE GAUSSIAN CLASSIFIER -----
    parameters = compute_parameters_MVG(DTR, LTR)   # calcolo i parametri per il train set

    # 1 - COMPUTE THE LIKELIHOOD
    S = class_conditional_prob(DVAL, parameters)

    print(S)
    # 2 - COMPUTE THE POSTERIOR PROBABILITY
    prior_prob = mcol(numpy.ones(3)/3.)   # ottengo un vettore colonna di 3 elementi contenenti 1/3, che indica la prior probability di ciascuna classe

    SJoint_prob = S + numpy.log(prior_prob)
    #print('---------my joint density: ------------\n', SJoint_prob, '\n')

    marginal_densities = mrow(scipy.special.logsumexp(SJoint_prob, axis=0))

    SPost = SJoint_prob - marginal_densities   # faccio la sottrazione dato che ho da calcolare il logaritmo di una divisione e ho gia' calcolato il logaritmo degli operandi


    #print('\n-----POSTERIOR PROBABILITY-----\n', SPost)

    predicted_val = SPost.argmax(0)
    print('predicted value: ', predicted_val)
    print('MVG - Errors: \n', predicted_val-LVAL)
    print('MVG - Error rate: ', (predicted_val-LVAL).sum()/float(LVAL.size)*100, '%')

    # ----- NAIVE BAYES -----
