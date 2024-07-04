import matplotlib
import matplotlib.pyplot
import numpy
import scipy
import scipy.optimize
import functions
import loadData
import plots


def trainLogReg(DTR, LTR, l):

    ZTR = LTR * 2.0 - 1     # sarebbe la z che moltiplica s nella sommatoria

    def logreg_obj(v):  # contiene i parametri per la logistic regression ==> v = [w, b]

        w, b = v[0: -1], v[-1]  # spacchetto i parametri    
        s = (functions.mcol(w).T @ DTR).ravel() + b     # s (separation surface) - ravel serve a ottenere il vettore monodimensionale

        loss = numpy.logaddexp(0, -ZTR * s)      # log (1 + e^-z*s) = log (e^0 - e^-z*s) = logsumexp(0, -z*s)

        reg_term = (l/2)*(numpy.linalg.norm(w)**2)  # lambda/2 * ||w||

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        GW = (functions.mrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()

        objective_function = reg_term + loss.mean()

        return objective_function, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(func = logreg_obj, x0 = numpy.zeros(DTR.shape[0]+1))[0] # cerco il minimo della funzione obbiettivo
    print('Log-reg - lambda =', l, ' -J(w, b) =', logreg_obj(vf)[0])
    return vf[:-1], vf[-1]  # ritorno w e b

def trainLogRegNonRegularized(DTR, LTR, l):

    ZTR = LTR * 2.0 - 1     # sarebbe la z che moltiplica s nella sommatoria

    def logreg_obj(v):  # contiene i parametri per la logistic regression ==> v = [w, b]

        w, b = v[0: -1], v[-1]  # spacchetto i parametri    
        s = (functions.mcol(w).T @ DTR).ravel() + b     # s (separation surface) - ravel serve a ottenere il vettore monodimensionale

        loss = numpy.logaddexp(0, -ZTR * s)      # log (1 + e^-z*s) = log (e^0 - e^-z*s) = logsumexp(0, -z*s)

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        GW = (functions.mrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()

        objective_function = loss.mean()

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
        s = (functions.mcol(w).T @ DTR).ravel() + b     # s (separation surface) - ravel serve a ottenere il vettore monodimensionale

        loss = numpy.logaddexp(0, -ZTR * s)      # log (1 + e^-z*s) = log (e^0 - e^-z*s) = logsumexp(0, -z*s)

        loss[ZTR > 0] *= wT
        loss[ZTR < 0] *= wF

        reg_term = (l/2)*(numpy.linalg.norm(w)**2)  # lambda/2 * ||w||

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        G[ZTR > 0] *= wT
        G[ZTR < 0] *= wF
        GW = (functions.mrow(G) * DTR).sum(1) + l * w.ravel()
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


def quadratic_feature_expansion(D):
    num_features, num_samples = D.shape
    expanded_features = []

    
    for i in range(num_samples):    # Compute the ourter product and flatten it (vec(x * x^T))
        x = D[:, i]
        outer_product = numpy.outer(x, x).reshape(-1, 1)
        expanded_features.append(numpy.vstack((outer_product, x.reshape(-1, 1))))

    expanded_features = numpy.hstack(expanded_features)   # Concatenate all expanded features as columns

    return expanded_features




########################################################
#                                                      #
#-------------------------MAIN-------------------------#
#                                                      #
########################################################
if __name__ == '__main__':

    D, L = loadData.load('trainData.txt')      # get the data and labels from the dataset
    (DTR, LTR), (DVAL, LVAL) = functions.split_training_test_dataset(D, L)

    lambda_values = numpy.logspace(-4, 2, 13)
    min_DCFs = []
    act_DCFs = []
    pT = 0.1
    
    for _lambda in lambda_values:
        w, b = trainLogReg(DTR, LTR, _lambda)   # calcolo i parametri del modello, w e b
        Sval = w.T @ DVAL + b       

        emp_prior = (LTR == 1).sum() / float(LTR.size)
        Sllr = Sval - numpy.log(emp_prior / (1-emp_prior))     
        DCF_min = compute_minDCF(Sllr, LVAL, 0.1, 1.0, 1.0)
        DCF_act = compute_actualDCF(Sllr, LVAL, 0.1, 1.0, 1.0)
        min_DCFs.append(DCF_min)
        act_DCFs.append(DCF_act)
        print('minDCF - pT = 0.1:', round(DCF_min,4))
        print('actDCF - pT = 0.1:', round(DCF_act,4))
        print()
        


    #plots.plot_lab8(min_DCFs, act_DCFs, lambda_values)

    # --- REDUCED TRAINING SET ---
    print('-'*40)
    print('REDUCED TRAINING SET')
    DTR_reduced = DTR[:, ::50]
    LTR_reduced = LTR[::50]

    lambda_values = numpy.logspace(-4, 2, 13)
    min_DCFs = []
    act_DCFs = []
    
    for _lambda in lambda_values:
        w, b = trainLogReg(DTR_reduced, LTR_reduced, _lambda)   # train the model

        Sval = w.T @ DVAL + b       
        emp_prior = (LTR == 1).sum() / float(LTR.size)
        Sllr = Sval - numpy.log(emp_prior / (1-emp_prior))     

        DCF_min = compute_minDCF(Sllr, LVAL, 0.1, 1.0, 1.0)
        DCF_act = compute_actualDCF(Sllr, LVAL, 0.1, 1.0, 1.0)
        min_DCFs.append(DCF_min)
        act_DCFs.append(DCF_act)
        print('minDCF - pT = 0.1:', round(DCF_min,4))
        print('actDCF - pT = 0.1:', round(DCF_act,4))
        print()
        


    #plots.plot_lab8(min_DCFs, act_DCFs, lambda_values)


    # --- WEIGHTED TRAINING SET ---
    print('-'*40)
    print('PRIOR-WEIGHTED MODEL - pT = 0.1')
    min_DCFs = []
    act_DCFs = []
    for _lambda in lambda_values:

        w, b = trainWeightedLoReg(DTR, LTR, _lambda, pT)
        Sval = w.T @ DVAL + b       # calcolo l'array di score
        Sllr = Sval - numpy.log(pT / (1-pT))     # rimuovo l'empirical prior dagli score per avere uno score application-dependant che si comporti come una llr

        DCF_min = compute_minDCF(Sllr, LVAL, pT, 1.0, 1.0)
        DCF_act = compute_actualDCF(Sllr, LVAL, pT, 1.0, 1.0)
        print('minDCF - pT = 0.1:', round(DCF_min,4))
        print('actDCF - pT = 0.1:', round(DCF_act,4))
        print()
        min_DCFs.append(DCF_min)
        act_DCFs.append(DCF_act)

    #plots.plot_lab8(min_DCFs, act_DCFs, lambda_values)



    # --- QUADRATIC LOGISTIC REGRESSION MODEL ---
    print('-'*40)
    print('QUADRATIC LOGISTIC REGRESSION MODEL')

    # Expand the training and validation features
    DTR_expanded = quadratic_feature_expansion(DTR)
    DVAL_expanded = quadratic_feature_expansion(DVAL)

    lambda_values = numpy.logspace(-4, 2, 13)
    print('lambda_values:', lambda_values)
    print('-'*40)
    min_DCFs = []
    act_DCFs = []
    pT = 0.1
    
    for _lambda in lambda_values:
        w, b = trainLogReg(DTR_expanded, LTR, _lambda)   # calcolo i parametri del modello, w e b
        Sval = w.T @ DVAL_expanded + b       

        emp_prior = (LTR == 1).sum() / float(LTR.size)
        Sllr = Sval - numpy.log(emp_prior / (1-emp_prior))     
        DCF_min = compute_minDCF(Sllr, LVAL, 0.1, 1.0, 1.0)
        DCF_act = compute_actualDCF(Sllr, LVAL, 0.1, 1.0, 1.0)
        min_DCFs.append(DCF_min)
        act_DCFs.append(DCF_act)
        print('minDCF - pT = 0.1:', round(DCF_min,4))
        print('actDCF - pT = 0.1:', round(DCF_act,4))
        print()

    #plots.plot_lab8(min_DCFs, act_DCFs, lambda_values)

    # --- CENTERING THE DATA ---
    print('-'*40)
    print('CENTERED DATA LOGISTIC REGRESSION')
    mean, _ = functions.compute_mean_covariance(DTR)
    DTR_centered = DTR - mean
    DVAL_centered = DVAL - mean

    lambda_values = numpy.logspace(-4, 2, 13)
    min_DCFs = []
    act_DCFs = []
    pT = 0.1
    
    for _lambda in lambda_values:
        w, b = trainLogReg(DTR_centered, LTR, _lambda)   # calcolo i parametri del modello, w e b
        Sval = w.T @ DVAL_centered + b       

        emp_prior = (LTR == 1).sum() / float(LTR.size)
        Sllr = Sval - numpy.log(emp_prior / (1-emp_prior))     
        DCF_min = compute_minDCF(Sllr, LVAL, 0.1, 1.0, 1.0)
        DCF_act = compute_actualDCF(Sllr, LVAL, 0.1, 1.0, 1.0)
        min_DCFs.append(DCF_min)
        act_DCFs.append(DCF_act)
        print('minDCF - pT = 0.1:', round(DCF_min,4))
        print('actDCF - pT = 0.1:', round(DCF_act,4))
        print()

    #plots.plot_lab8(min_DCFs, act_DCFs, lambda_values)

    min_DCFs = []
    act_DCFs = []
    pT = 0.1
    print('-'*40)
    print('NON REGULARIZED CENTERED LOG REG')
    for _lambda in lambda_values:
        w, b = trainLogRegNonRegularized(DTR_centered, LTR, _lambda)   # calcolo i parametri del modello, w e b
        Sval = w.T @ DVAL_centered + b       

        emp_prior = (LTR == 1).sum() / float(LTR.size)
        Sllr = Sval - numpy.log(emp_prior / (1-emp_prior))     
        DCF_min = compute_minDCF(Sllr, LVAL, 0.1, 1.0, 1.0)
        DCF_act = compute_actualDCF(Sllr, LVAL, 0.1, 1.0, 1.0)
        min_DCFs.append(DCF_min)
        act_DCFs.append(DCF_act)
        print('minDCF - pT = 0.1:', round(DCF_min,4))
        print('actDCF - pT = 0.1:', round(DCF_act,4))
        print()

    #plots.plot_lab8(min_DCFs, act_DCFs, lambda_values)



