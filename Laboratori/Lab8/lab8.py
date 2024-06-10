import matplotlib
import matplotlib.pyplot
import numpy
import scipy
import scipy.optimize
import sklearn.datasets

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))


def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]
    L = L[L != 0]
    L[L==2] = 0

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

def f(x):   # x ha 2 righe: la prima contiene la y, la seconda la z

    y = x[0]    
    z = x[1]

    f = (y + 3)**2 + numpy.sin(y) + (z + 1)**2

    return f

def fprime(x):   # x ha 2 righe: la prima contiene la y, la seconda la z

    y = x[0]    
    z = x[1]

    f = numpy.array([2*(y + 3) + numpy.cos(y), 2*(z+1)])

    return f


def trainLogReg(DTR, LTR, l):

    ZTR = LTR * 2.0 - 1     # sarebbe la z che moltiplica s nella sommatoria

    def logreg_obj(v):  # contiene i parametri per la logistic regression ==> v = [w, b]

        w, b = v[0: -1], v[-1]  # spacchetto i parametri    
        s = (mcol(w).T @ DTR).ravel() + b     # s (separation surface) - ravel serve a ottenere il vettore monodimensionale

        loss = numpy.logaddexp(0, -ZTR * s)      # log (1 + e^-z*s) = log (e^0 - e^-z*s) = logsumexp(0, -z*s)

        reg_term = (l/2)*(numpy.linalg.norm(w)**2)  # lambda/2 * ||w||

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        GW = (mrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()

        objective_function = reg_term + loss.mean()

        return objective_function, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(func = logreg_obj, x0 = numpy.zeros(DTR.shape[0]+1))[0] # cerco il minimo della funzione obbiettivo
    print('Log-reg - lambda =', l, ' -J(w, b) =', logreg_obj(vf)[0])
    return vf[:-1], vf[-1]  # ritorno w e b


def trainWeightedLoReg(DTR, LTR, l, pT):

    ZTR = LTR * 2.0 - 1     # sarebbe la z che moltiplica s nella sommatoria

    wT = pT / (ZTR > 0).sum()
    wF = (1 - pT) / (ZTR < 0).sum()

    def logreg_obj(v):  # contiene i parametri per la logistic regression ==> v = [w, b]

        w, b = v[0: -1], v[-1]  # spacchetto i parametri    
        s = (mcol(w).T @ DTR).ravel() + b     # s (separation surface) - ravel serve a ottenere il vettore monodimensionale

        loss = numpy.logaddexp(0, -ZTR * s)      # log (1 + e^-z*s) = log (e^0 - e^-z*s) = logsumexp(0, -z*s)

        loss[ZTR > 0] *= wT
        loss[ZTR < 0] *= wF

        reg_term = (l/2)*(numpy.linalg.norm(w)**2)  # lambda/2 * ||w||

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        G[ZTR > 0] *= wT
        G[ZTR < 0] *= wF
        GW = (mrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()

        objective_function = reg_term + loss.sum()

        return objective_function, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(func = logreg_obj, x0 = numpy.zeros(DTR.shape[0]+1))[0] # cerco il minimo della funzione obbiettivo
    print('weighted Log-reg - lambda =', l, ' -J(w, b) =', logreg_obj(vf)[0])
    return vf[:-1], vf[-1]  # ritorno w e b


def compute_minDCF(llr, labels, prior, Cfn, Cfp, returnThreshold=False):

    llr_sorted = llr

    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), llr_sorted, numpy.array([numpy.inf])])
    DCF_min = None
    DCF_th = None

    for th in thresholds:

        predicted_labels = numpy.int32(llr > th)
        conf_matrix = compute_confusion_matrix(predicted_labels, labels)
        DCF_normalized = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp, normalize=True)

        if DCF_min is None or DCF_normalized < DCF_min:
            DCF_min = DCF_normalized
            DCF_th = th

    if returnThreshold:
        return DCF_min, DCF_th
    
    return DCF_min

def compute_confusion_matrix(P, L):

    nclasses = L.max() + 1
    conf_mat = numpy.zeros((nclasses, nclasses), dtype = numpy.int32)

    for i in range(P.shape[0]):    # uso i valori contenuti in P ed L come indici per creare la confusion matrix

        index_pred = P[i]   # prendo il valore contenuto in P
        index_class = L[i]  # prendo il valore contenuto in L

        conf_mat[index_pred][index_class] += 1

    return conf_mat

def compute_empirical_bayes_risk_binary(confusion_matrix, prior, Cfn, Cfp, normalize=True):
    Pfn = confusion_matrix[0][1] / (confusion_matrix[0][1] + confusion_matrix[1][1])   # false negative rate
    Pfp = confusion_matrix[1][0] / (confusion_matrix[1][0] + confusion_matrix[0][0])   # false positive rate
    DCFu = (prior*Cfn*Pfn) + (1 - prior)*Cfp*Pfp

    if normalize:
        return DCFu / numpy.minimum(prior * Cfn, (1-prior)*Cfp)

    return DCFu

def compute_minDCF(llr, labels, prior, Cfn, Cfp, returnThreshold=False):

    llr_sorted = llr

    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), llr_sorted, numpy.array([numpy.inf])])
    DCF_min = None
    DCF_th = None

    for th in thresholds:

        predicted_labels = numpy.int32(llr > th)
        conf_matrix = compute_confusion_matrix(predicted_labels, labels)
        DCF_normalized = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp, normalize=True)

        if DCF_min is None or DCF_normalized < DCF_min:
            DCF_min = DCF_normalized
            DCF_th = th

    if returnThreshold:
        return DCF_min, DCF_th
    
    return DCF_min

def compute_optimal_bayes_binary_llr(llr, prior, Cfn, Cfp):
    threshold = -numpy.log((prior * Cfn) / ((1-prior)*Cfp))   # uso la formula per calcolare la threshold e seleziono poi i llr > threshold

    return numpy.int32(llr > threshold)

def compute_actualDCF(llr, labels, prior, Cfn, Cfp, normalize=True):

    predictions = compute_optimal_bayes_binary_llr(llr, prior, Cfn, Cfp)

    conf_matrix = compute_confusion_matrix(predictions, labels)

    return compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp, normalize)





########################################################
#                                                      #
#-------------------------MAIN-------------------------#
#                                                      #
########################################################
if __name__ == '__main__':

    D, L = load_iris_binary()      # carico il dataset
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)      # splitto il dataset in training e evaluation

    #print(scipy.optimize.fmin_l_bfgs_b(func = f, approx_grad = True, x0 = numpy.zeros(2)))
    #print(scipy.optimize.fmin_l_bfgs_b(func = f, fprime = fprime, x0 = numpy.zeros(2)))


    for _lambda in [1e-3, 1e-1, 1.0]:
        
        w, b = trainLogReg(DTR, LTR, _lambda)   # calcolo i parametri del modello, w e b
        Sval = w.T @ DVAL + b       # calcolo l'array di score
        PVAL = (Sval > 0)*1   # assegno 1 e 0 per assegnare la classe, in base alla threshold, in questo caso di 0
        err_rate = ((PVAL != LVAL).sum() / float(LVAL.size))*100  # calcolo error rate

        print('error rate:', round(err_rate,3),'%')

        emp_prior = (LTR == 1).sum() / float(LTR.size)
        Sllr = Sval - numpy.log(emp_prior / (1-emp_prior))     # rimuovo l'empirical prior dagli score per avere uno score application-dependant che si comporti come una llr

        DCF_min = compute_minDCF(Sllr, LVAL, 0.5, 1.0, 1.0)
        DCF_act = compute_actualDCF(Sllr, LVAL, 0.5, 1.0, 1.0)
        print('minDCF - pT = 0.5:', round(DCF_min,4))
        print('actDCF - pT = 0.5:', round(DCF_act,4))
        print()

        pT = 0.8
        w, b = trainWeightedLoReg(DTR, LTR, _lambda, pT)
        Sval = w.T @ DVAL + b       # calcolo l'array di score
        Sllr = Sval - numpy.log(pT / (1-pT))     # rimuovo l'empirical prior dagli score per avere uno score application-dependant che si comporti come una llr

        DCF_min = compute_minDCF(Sllr, LVAL, pT, 1.0, 1.0)
        DCF_act = compute_actualDCF(Sllr, LVAL, pT, 1.0, 1.0)
        print('minDCF - pT = 0.8:', round(DCF_min,4))
        print('actDCF - pT = 0.8:', round(DCF_act,4))
        print()
