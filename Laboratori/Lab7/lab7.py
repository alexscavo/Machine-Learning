import matplotlib
import matplotlib.pyplot
import numpy
import scipy
import load
import sklearn.datasets

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))

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

def compute_parameters_MVG(D, L):
    labels = set(L)     # ottengo le singole label
    parameters = {}     # creo il dict che conterra' i parametri
    for label in labels:    
        DX = D[:, L == label]   # considero solo i sample di classe label
        parameters[label] = compute_mean_covariance(DX)     # inserisco mu e C nel dict in posizione label

    return parameters


def class_conditional_prob(D, parameters):
    S = numpy.zeros((len(parameters), D.shape[1]))    

    

    for label in range(S.shape[0]):     # per ogni classe
        S[label, :] = logpdf_GAU_ND(D, parameters[label][0], parameters[label][1])  # calcolo la class conditional probability di ciascun sample data la classe in questione

    return S

def compute_parameters_tied(D, L):
    labels = set(L)
    parameters = {}
    means = {}
    C_global = 0
    for label in labels:
        DX = D[:, L == label]
        mu, C = compute_mean_covariance(DX)
        C_global += C * DX.shape[1]
        means[label] = mu

    SW = C_global / D.shape[1]

    for label in labels:
        parameters[label] = (means[label], SW)

    return parameters


# mi dice quanto e' probabile che ciascun sample di x appartenga alla classe di cui i parametri sono mu e C
def logpdf_GAU_ND(x, mu, C):
    
    M = x.shape[0]     # mu e' un array (M, 1)
    P = numpy.linalg.inv(C)     # inverse of the covariance matrix

    pdf = -(M/2)*numpy.log(2*numpy.pi) - 0.5 * numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

    return pdf

# funzione per calcolare la confusion matrix, partendo dalle predizioni fatte
def compute_confusion_matrix(P, L):

    nclasses = L.max() + 1
    conf_mat = numpy.zeros((nclasses, nclasses), dtype = int)

    for i in range(P.shape[0]):    # uso i valori contenuti in P ed L come indici per creare la confusion matrix

        index_pred = P[i]   # prendo il valore contenuto in P
        index_class = L[i]  # prendo il valore contenuto in L

        conf_mat[index_pred][index_class] += 1

    return conf_mat

def compute_dictionary(tercets):

    train_dict = set([])     # set di parole

    for tercet in tercets:  # per ogni terzina nel training set
        words = tercet.split()  # dovrebbe separare tutte le parole

        for word in words:
            train_dict.add(word)

    return train_dict

def sol1_estimate_model(tercets_train, eps = 0.1):
    

    common_dict = set([])   # dizionario comune a tute le classi per calcolare il numero totale di parole

    for cls in tercets_train:   # per ogni classe
        tercets_list = tercets_train[cls]   # prendo la lista di terzine
        class_dict = compute_dictionary(tercets_list)  # cerco tutte le parole del canto
        common_dict = common_dict.union(class_dict) # dizionario generale di tutti i canti

    words_per_class = {}    # inizializzo il dizionario di parole delle classi, che contera' per ciascuna parola il numero di occorrenze, per ciascun canto
    for cls in tercets_train:   # per ciascun canto
        words_per_class[cls] = {word: eps for word in common_dict}    # creo una entry nel dizionario avente come valore un dizionario le cui chiavi sono le parole presenti nel dizionario comune

    
    # calcolo le occorrenze delle parole in ciascun canto
    for cls in tercets_train:
        tercets_list = tercets_train[cls]   # prendo la lista di terzine

        for tercet in tercets_list: # per ogni terzina della lista
            words = tercet.split()      # suddivido le parole della terzina

            for word in words:  # per ogni parola della terzina
                words_per_class[cls][word] += 1     # incremento il suo conteggio
    
    # words frequencies for each cantica (class)
    for cls in tercets_train:
        words_num = sum(words_per_class[cls].values())  # conto il numero di occorrenze totali di tutte le parole del canto
        
        for word in words_per_class[cls]:
            words_per_class[cls][word] = numpy.log(words_per_class[cls][word]) - numpy.log(words_num)   # log(N_{c,j} / N_c)

    return words_per_class

# calcolo quanto la probabilità di ciasuna classe di essere la classe della terzina passata
def sol1_compute_ll(tercet, class_log_prob):

    # class0 = prob
    # class1 = prob
    # class2 = prob
    class_ll = {cls: 0 for cls in class_log_prob} # inizializzo la prob di ciascuna classe a 0   
    words = tercet.split()  # parole della terzina

    for cls in class_log_prob:  # per ogni classe del 
        for word in words: # per ogni paraola nella terzina
            if word in class_log_prob[cls]: # se la parola è tra le parole del dizionario della classe
                class_ll[cls] += class_log_prob[cls][word]    # aggiungo la prob della parola di appartenere alla classe cls

    return class_ll

def estimate_class_conditional_ll(tercets, class_log_prob, class_indeces = None):   # model contains the words frequencies for each class

    if class_indeces is None:
        class_indeces = {cls: index for index in enumerate(sorted(class_log_prob))}

    # matrice degli score, righe = classi/cantiche, colonne = sample/terzina
    S = numpy.zeros((len(class_log_prob), len(tercets)))    

    for tercet_index, tercet in enumerate(tercets):    # per ogni terzina in ingresso prendo la terzina e il suo indice
        class_scores = sol1_compute_ll(tercet, class_log_prob)     # calcolo la prob di ciascuna classe di essere la classe della terzina in questione

        # ora riempo la matrice degli score, inserendo nella posizione corretta la probabilità calcolata
        for cls in class_log_prob:  
            class_index = class_indeces[cls]    # prendo l'indice della classe. Mi serve dato che per ora le classi erano ancora stringhe
            S[class_index, tercet_index] = class_scores[cls]

    return S

def compute_posterior_prob(scores, prior_prob = None):

    if prior_prob is None:
        prior_prob = mcol(numpy.log( numpy.ones(scores.shape[0]) / float(scores.shape[0]) ))

    joint_prob = scores + prior_prob
    marginal_densities = scipy.special.logsumexp(joint_prob, axis=0)
    posterior_prob = joint_prob - marginal_densities
    return numpy.exp(posterior_prob)

# optimal Bayes decisions for binary tasks with log-likelihood-ratio scores
def compute_optimal_bayes_binary_llr(llr, prior, Cfn, Cfp):
    threshold = -numpy.log((prior * Cfn) / ((1-prior)*Cfp))   # uso la formula per calcolare la threshold e seleziono poi i llr > threshold

    return numpy.int32(llr > threshold)

def compute_bayes_risk_binary(confusion_matrix, prior, Cfn, Cfp):
    Pfn = confusion_matrix[0][1] / (confusion_matrix[0][1] + confusion_matrix[1][1])   # false negative rate
    Pfp = confusion_matrix[1][0] / (confusion_matrix[1][0] + confusion_matrix[0][0])   # false positive rate
    DCFu = (prior*Cfn*Pfn) + (1 - prior)*Cfp*Pfp

    return DCFu

def compute_normalized_DCF(DCF, prior, Cfn, Cfp):
    bayes_risk_dummy = numpy.min([prior*Cfn, (1-prior)*Cfp])    # bayes risk di un sistema che predice o sempre vero, o sempre falso

    norm_DCF = DCF / bayes_risk_dummy

    return norm_DCF


def compute_minDCF_slow(llr, labels, prior, Cfn, Cfp, returnThreshold=False):

    llr_sorted = llr

    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), llr_sorted, numpy.array([numpy.inf])])
    DCF_min = None
    DCF_th = None

    for th in thresholds:

        predicted_labels = numpy.int32(llr > th)
        conf_matrix = compute_confusion_matrix(predicted_labels, labels)
        DCF = compute_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp)
        DCF_normalized = compute_normalized_DCF(DCF, prior, Cfn, Cfp)

        if DCF_min is None or DCF_normalized < DCF_min:
            DCF_min = DCF_normalized
            DCF_th = th

    if returnThreshold:
        return DCF_min, DCF_th
    
    return DCF_min

# funzione che ritorna tutte le possibili combinazioni di Pfp e Pfn per ciascuna threshold (ovvero stiamo calcolando i punti della ROC)
def compute_Pfn_Pfp(llr, labels):
    
    llrSorter = numpy.argsort(llr)
    llrSorted = llr[llrSorter]  # ordiniamo llr in modo da avere tutti le labels in ordine
    classLabelSorted = labels[llrSorter] # con lo stesso ordinamento ordiniamo anche le label, in modo che continuino a combaciare con gli score

    Pfp = []
    Pfn = []

    nTrue = (classLabelSorted == 1).sum()   # quanti appartengono alla classe 1
    nFalse = (classLabelSorted == 0).sum()  # quanti appartengono alla classe 0
    nFalseNegative = 0  # nella prima iterazione, llr > th con la th piu' bassa di tutte, quindi sto assegnando tutti alla classe 1. Di conseguenza non sto sbagliando a classificare nessun positivo
    nFalsePositive = nFalse # tutti quelli della classe 0 sono stati assegnati alla classe 1. = a nFalse dato che siamo assegnando tutti alla classe 1 (inizialmente)

    Pfn.append(nFalseNegative / nTrue)  # FNR
    Pfp.append(nFalsePositive / nFalse) # FPR

    for index in range(len(llrSorted)):
        if classLabelSorted[index] == 1:
            nFalseNegative += 1
        if classLabelSorted[index] == 0:
            nFalsePositive -= 1

        Pfp.append(nFalsePositive / nTrue)
        Pfn.append(nFalseNegative / nFalse)

    llrSorted = numpy.concatenate([-numpy.array([numpy.inf]), llrSorted])   # aggiungo -inf alla fine degli score

    # se alcuni valori degli score fossero uguali, abbiamo delle thresholds ripetute, di conseguenza dobbiamo compattare Pfn e Pfp
    
    PfnOut = []
    PfpOut = []
    thresholdsOut = []
    for index in range(len(llrSorted)):

        if index == len(llrSorted) - 1 or llrSorted[index+1] != llrSorted[index]:
            PfnOut.append(Pfn[index])
            PfpOut.append(Pfp[index])
            thresholdsOut.append(llrSorted[index])

    return numpy.array(PfnOut), numpy.array(PfpOut), numpy.array(thresholdsOut)
 
# copmute the matrix of posteriors from class-conditional log-likelihoods and prior array
def compute_posteriors(log_class_conditional_ll, prior):
    joint_prob = log_class_conditional_ll + prior_prob
    marginal_densities = scipy.special.logsumexp(joint_prob, axis=0)
    posterior_prob = joint_prob - marginal_densities
    return numpy.exp(posterior_prob)

#compute optimal Bayes decisions for the matrix of class posteriors
def compute_optimal_Bayes(posterior, cost_matrix):
    expected_costs = cost_matrix @ posterior
    return numpy.argmin(expected_costs, axis = 0)

def compute_empirical_bayes_risk(conf_matrix, prior_prob, cost_matrix, normalize = True):

    error_rates = conf_matrix / mrow(conf_matrix.sum(0))   # faccio la somma degli elementi per una colonna per sapere quante terzine di quella classe ci sono e ottenere l'error rate corrispondente per quelle predizioni
    
    bayes_error = ((error_rates * cost_matrix).sum(0) * mrow(prior_prob).ravel()).sum()

    if normalize:
        return bayes_error / numpy.min(cost_matrix @ prior_prob)
    
    return bayes_error


















########################################################
#                                                      #
#-------------------------MAIN-------------------------#
#                                                      #
########################################################
if __name__ == '__main__':

    D, L = load_iris()      # carico il dataset
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)      # splitto il dataset in training e evaluation

    # --- MVG ---
    parameters = compute_parameters_MVG(DTR, LTR)   # calcolo i parametri per il train set
    S = class_conditional_prob(DVAL, parameters)
    prior_prob = mcol(numpy.ones(3)/3.)   # ottengo un vettore colonna di 3 elementi contenenti 1/3, che indica la prior probability di ciascuna classe
    SJoint_prob = S + numpy.log(prior_prob)
    marginal_densities = mrow(scipy.special.logsumexp(SJoint_prob, axis=0))
    SPost = SJoint_prob - marginal_densities   # faccio la sottrazione dato che ho da calcolare il logaritmo di una divisione e ho gia' calcolato il logaritmo degli operandi
    predicted_val = SPost.argmax(0)
    conf_mat = compute_confusion_matrix(predicted_val, LVAL)
    print('Confusion matrix for the MVG classifier:\n',conf_mat)

    # --- tied covariance classifier ---
    parameters_tied = compute_parameters_tied(DTR, LTR)
    S = class_conditional_prob(DVAL, parameters_tied)
    prior_prob = mcol(numpy.ones(3)/3.)  
    SJoint_prob = S + numpy.log(prior_prob)
    marginal_densities = mrow(scipy.special.logsumexp(SJoint_prob, axis=0))
    SPost = SJoint_prob - marginal_densities   
    predicted_val = SPost.argmax(0)
    conf_mat = compute_confusion_matrix(predicted_val, LVAL)
    print('\nconfusion matrix for the tied covariance classifier:\n', conf_mat)




    #
    # ----- SOLUZIONE 1 -----
    #
    # carico il dataset
    lInf, lPur, lPar = load.load_data()

    # splitto i dataset per ottenere training e validation (25% validation)
    lInf_train, lInf_evaluation = load.split_data(lInf, 4)
    lPur_train, lPur_evaluation = load.split_data(lPur, 4)
    lPar_train, lPar_evaluation = load.split_data(lPar, 4)

    # prior probabilities
    prior_prob = numpy.log(mcol(numpy.ones(3)/.3))

    class_to_index = {'inferno': 0, 'purgatorio': 1,  'paradiso': 2}    # serve per il mapping delle parole e con gli indici

    tercets_train = {       # dizionario con chiave la classe e come valore la lista di terzine di quella classe
        'inferno': lInf_train,  
        'purgatorio': lPur_train,
        'paradiso': lPar_train
    }

    tercets_eval = lInf_evaluation + lPur_evaluation + lPar_evaluation  # l'evaluation lo faccio su tutti gli evaluation set

    sol1_model = sol1_estimate_model(tercets_train, eps = 0.001)

    # predicting the cantica
    scores = estimate_class_conditional_ll(tercets_eval, sol1_model, class_to_index)    # ottengo la matrice che per ogni ogni terzina indica la probabilità che appartenga alle 3 classi

    # analyze performances
    posterior_prob = compute_posterior_prob(scores, prior_prob) # calcolo la posterior prob di ciascuna terzina per ciascuna classe

    labelsInf = numpy.zeros(len(lInf_evaluation), dtype = int) # inizializzo il vettore perchè non potrei altrimenti assegnarvi valori
    labelsInf[:] = class_to_index['inferno']    # assegno a tutti i valori del vettore la classe equivalente ad 'inferno'

    labelsPur = numpy.zeros(len(lPur_evaluation), dtype = int) # inizializzo il vettore perchè non potrei altrimenti assegnarvi valori
    labelsPur[:] = class_to_index['purgatorio']    # assegno a tutti i valori del vettore la classe equivalente ad 'inferno'

    labelsPar = numpy.zeros(len(lPar_evaluation), dtype = int) # inizializzo il vettore perchè non potrei altrimenti assegnarvi valori
    labelsPar[:] = class_to_index['paradiso']    # assegno a tutti i valori del vettore la classe equivalente ad 'inferno'

    labels_eval = numpy.hstack([labelsInf, labelsPur, labelsPar])    # impilo i 3 vettori di labels
    predicted_labels = numpy.argmax(posterior_prob, axis = 0)

    conf_matrix = compute_confusion_matrix(predicted_labels, labels_eval)
    print('-'*40)
    print('confusion matrix for the divina commedia dataset:\n', conf_matrix)

    llr_commedia = numpy.load('Lab7\commedia_llr_infpar.npy')   
    labels_commedia = numpy.load('Lab7\commedia_labels_infpar.npy')

    for prior, Cfn, Cfp in [(0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]:  # per ciascuna tripletta indica la prior prob, the cost of false negatives and the one of false positives
        print()
        print('Prior:', prior, '- Cfn:', Cfn, '- Cfp:', Cfp)

        predictions_binary = compute_optimal_bayes_binary_llr(llr_commedia, prior, Cfn, Cfp)    # predizioni fatte dal classificatore R
        conf_matrix = compute_confusion_matrix(predictions_binary, labels_commedia)
        print(conf_matrix)

        # --- BINARY TASK: EVALUATION ---
        # in questo modo possiamo valuare il bayes risk, che indica il costo che paghiamo quando effettuiamo le decisioni c* (usando la conf_matrix) per i dati di test
        # ci permette di confrontare i vari sistemi, ma non ci indica i benefici dell'usare il nostro recognizer rispetto alla optimal bayes decision che si basa solamente sulle prior info
        bayes_risk = compute_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp)   
        print('Bayes risk:', round(bayes_risk, 3))

        # possiamo quindi calcolare un detection cost normalizzato, dividendo il bayes risk per il rischio di un ipotetico sistema che non usa i dati di test (dummy system)
        normalized_bayes = compute_normalized_DCF(bayes_risk, prior, Cfn, Cfp)
        print('normalized Bayes risk:', round(normalized_bayes, 3)) # notiamo che solo in 2 casi il DCF normalizzato e' sotto l'1, negli altri casi e' dannoso


        # --- MINIMUM DETECTION COST ---
        # dato che il nostro classificatore non produce in output i log-likelihood ratio, la threshold ottima non e' piu' valida e di conseguenza facciamo fatica a definire i costi dovuti a
        # una scarsa separazione di classe e quelli dovuti ad una scarsa calibrazione. Questo quindi viene definito come score mis-calibrated.
        # Possiamo quindi ricalibrare gli score usando un piccolo set di sample con classificati (un validation set, che ci permette di trovare i parametri per migliorare il nostro classificatore)
        # Alternativamente possiamo calcolare la threshold ottima sul validation set, usandola poi per il test set.
        # Per fare questo possiamo ad esempio calcolare la DCF normalizzata sul test set usando tutte le possibili thresholds, selezionando poi il valore minimo. In questo modo
        # troviamo un lower bound per la DCF che il sistema puo' raggiungere

        DCF_min, threshold_min = compute_minDCF_slow(llr_commedia, labels_commedia, prior, Cfn, Cfp, True)
        print('DCF min:', round(DCF_min, 3), '- Threshold:', round(threshold_min, 3))
        # possiamo notare come ad esclusione della prima, tutte le altre DCF indichino una perdita dovuta ad una scarsa calibrazione, in particolare per le ultime due applicazioni
        # che erano quelle con DCF normalizzata > 1. Quindi quello che accade in questi due casi e' che noi otteniamo comunque degli score e che li utilizziamo per fare delle decisioni
        # ma le decisioni, ma non sapevamo utilizzarli per fare delle predizioni piu' accurate di quanto non avremmo saputo fare con le sole prior info

    # --- ROC CURVES ---
    # servono per indicare il trade-off tra i vari tipi di errore. Noi vedremo, dato che e' la piu' usata, quella che mappa true positives vs. false positives. 
    # Una ROC ideale equivarrebbe ad una funzione gradino, dato che massimizzerei il true positive rate e minimizzerei il false positive rate, mentre una ROC lineare "diagonale"
    # indicherebbe un random classifier (ovvero dal quale puo' uscire qualsiasi cosa, dato che assegna le label in modo random). 
    # Ciascun punto della ROC e' (Ptp(t), Pfp(t)), dove Ptp e' il true positive rate, mentre Pfp e' il false positive rate (Ptp = 1 - Pfp). La t invece indica la threshold
    
    print('-'*40)
    Pfn, Pfp, _ = compute_Pfn_Pfp(llr_commedia, labels_commedia)
    Ptp = 1 - Pfn
    '''matplotlib.pyplot.figure(0)
    matplotlib.pyplot.title('ROC curve: TPR - FPR')
    matplotlib.pyplot.xlabel('FPR')
    matplotlib.pyplot.ylabel('TPR')
    matplotlib.pyplot.plot(Pfp, Ptp)    # vuole prima le x poi le y
    matplotlib.pyplot.grid()
    matplotlib.pyplot.show()'''
    

    # --- BAYES ERROR PLOT ---
    # tilde_p = function of prior log-odds
    # tilde_pi = effective prior
    effPriorLogOdds = numpy.linspace(-3, 3, 21)     # creo una serie di punti equispaziati (21, dato che e' il numero di punti che valutiamo con la DCF)
    effPriors = 1.0 / (1.0 + numpy.exp(-effPriorLogOdds))

    actual_DCF = []
    min_DCF = []

    for effPrior in effPriors:
        # le triplette ora sono (tilde_pi, 1, 1)
        predictions = compute_optimal_bayes_binary_llr(llr_commedia, effPrior, 1.0, 1.0)
        conf_matrix = compute_confusion_matrix(predictions, labels_commedia)

        bayes_risk = compute_bayes_risk_binary(conf_matrix, effPrior, 1.0, 1.0)    # DCF
        DCF = compute_normalized_DCF(bayes_risk, effPrior, 1.0, 1.0)    # NORMALIZED DCF
        minDCF = compute_minDCF_slow(llr_commedia, labels_commedia, effPrior, 1.0, 1.0)    # MIN DCF

        actual_DCF.append(DCF)
        min_DCF.append(minDCF)

    '''matplotlib.pyplot.plot(effPriorLogOdds, actual_DCF, label='actual DCF eps 0.001', color = 'r')
    matplotlib.pyplot.plot(effPriorLogOdds, min_DCF, label='DCF eps 0.001', color = 'b')
    matplotlib.pyplot.ylim([0, 1.1])
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel('prior log-odds')
    matplotlib.pyplot.ylabel('DCF value')
    #matplotlib.pyplot.show()'''
    
    llr_commedia = numpy.load('Lab7\commedia_llr_infpar_eps1.npy')   
    labels_commedia = numpy.load('Lab7\commedia_labels_infpar_eps1.npy')

    print('-'*40)
    for prior, Cfn, Cfp in [(0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]:  # per ciascuna tripletta indica la prior prob, the cost of false negatives and the one of false positives
        print()
        print('Prior:', prior, '- Cfn:', Cfn, '- Cfp:', Cfp)

        predictions_binary = compute_optimal_bayes_binary_llr(llr_commedia, prior, Cfn, Cfp)    # predizioni fatte dal classificatore R
        conf_matrix = compute_confusion_matrix(predictions_binary, labels_commedia)       
        bayes_risk = compute_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp)   
        print('Bayes risk:', round(bayes_risk, 3))
        # possiamo quindi calcolare un detection cost normalizzato, dividendo il bayes risk per il rischio di un ipotetico sistema che non usa i dati di test (dummy system)
        normalized_bayes = compute_normalized_DCF(bayes_risk, prior, Cfn, Cfp)
        print('normalized Bayes risk:', round(normalized_bayes, 3)) # notiamo che solo in 2 casi il DCF normalizzato e' sotto l'1, negli altri casi e' dannoso
        DCF_min, threshold_min = compute_minDCF_slow(llr_commedia, labels_commedia, prior, Cfn, Cfp, True)
        print('DCF min:', round(DCF_min, 3), '- Threshold:', round(threshold_min, 3))
    
    effPriorLogOdds = numpy.linspace(-3, 3, 21)     # creo una serie di punti equispaziati (21, dato che e' il numero di punti che valutiamo con la DCF)
    effPriors = 1.0 / (1.0 + numpy.exp(-effPriorLogOdds))

    actual_DCF = []
    min_DCF = []

    for effPrior in effPriors:
        # le triplette ora sono (tilde_pi, 1, 1)
        predictions = compute_optimal_bayes_binary_llr(llr_commedia, effPrior, 1.0, 1.0)
        conf_matrix = compute_confusion_matrix(predictions, labels_commedia)

        bayes_risk = compute_bayes_risk_binary(conf_matrix, effPrior, 1.0, 1.0)    # DCF
        DCF = compute_normalized_DCF(bayes_risk, effPrior, 1.0, 1.0)    # NORMALIZED DCF
        minDCF = compute_minDCF_slow(llr_commedia, labels_commedia, effPrior, 1.0, 1.0)    # MIN DCF

        actual_DCF.append(DCF)
        min_DCF.append(minDCF)

    '''matplotlib.pyplot.plot(effPriorLogOdds, actual_DCF, label='actual DCF eps 1.0', color = 'y')
    matplotlib.pyplot.plot(effPriorLogOdds, min_DCF, label='DCF eps 1.0', color = 'c')
    matplotlib.pyplot.ylim([0, 1.1])
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel('prior log-odds')
    matplotlib.pyplot.ylabel('DCF value')
    matplotlib.pyplot.show()
'''





    #
    # --- MULTICLASS EVALUATION ---
    #
    print()
    print('-'*40)
    print()
    print('Multiclass task')

    C = numpy.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    prior_prob = mcol(numpy.log(numpy.array([0.3, 0.4, 0.3])))

    print()
    print('Eps 0.001')
    commedia_ll = numpy.load('Lab7\commedia_ll.npy')
    commedia_labels = numpy.load('Lab7\commedia_labels.npy')

    posterior_probs = compute_posteriors(commedia_ll, prior_prob)
    commedia_predictions = compute_optimal_Bayes(posterior_probs, C)
    conf_matrix = compute_confusion_matrix(commedia_predictions, commedia_labels)
    print('Confusion matrix:')
    print(conf_matrix)
    emp_bayes_risk = compute_empirical_bayes_risk(conf_matrix, numpy.exp(prior_prob), C, normalize=False)
    print('Empirical Bayes risk:', round(emp_bayes_risk, 3))
    normalized_bayes_risk = compute_empirical_bayes_risk(conf_matrix, numpy.exp(prior_prob), C)
    print('Normalized empirical Bayes risk:', round(normalized_bayes_risk, 3))

    print()
    print('Eps 1.0')
    commedia_ll = numpy.load('Lab7\commedia_ll_eps1.npy')
    commedia_labels = numpy.load('Lab7\commedia_labels_eps1.npy')

    posterior_probs = compute_posteriors(commedia_ll, prior_prob)
    commedia_predictions = compute_optimal_Bayes(posterior_probs, C)
    conf_matrix = compute_confusion_matrix(commedia_predictions, commedia_labels)
    print('Confusion matrix:')
    print(conf_matrix)
    emp_bayes_risk = compute_empirical_bayes_risk(conf_matrix, numpy.exp(prior_prob), C, normalize=False)
    print('Empirical Bayes risk:', round(emp_bayes_risk, 3))
    normalized_bayes_risk = compute_empirical_bayes_risk(conf_matrix, numpy.exp(prior_prob), C)
    print('Normalized empirical Bayes risk:', round(normalized_bayes_risk, 3))




    C = numpy.ones((3, 3)) - numpy.eye(3)
    prior_prob = numpy.log(mcol(numpy.ones(3)/3.0))
    print()
    print('Eps 0.001')
    commedia_ll = numpy.load('Lab7\commedia_ll.npy')
    commedia_labels = numpy.load('Lab7\commedia_labels.npy')

    posterior_probs = compute_posteriors(commedia_ll, prior_prob)
    commedia_predictions = compute_optimal_Bayes(posterior_probs, C)
    conf_matrix = compute_confusion_matrix(commedia_predictions, commedia_labels)
    print('Confusion matrix:')
    print(conf_matrix)
    emp_bayes_risk = compute_empirical_bayes_risk(conf_matrix, numpy.exp(prior_prob), C, normalize=False)
    print('Empirical Bayes risk:', round(emp_bayes_risk, 3))
    normalized_bayes_risk = compute_empirical_bayes_risk(conf_matrix, numpy.exp(prior_prob), C)
    print('Normalized empirical Bayes risk:', round(normalized_bayes_risk, 3))

    print()
    print('Eps 1.0')
    commedia_ll = numpy.load('Lab7\commedia_ll_eps1.npy')
    commedia_labels = numpy.load('Lab7\commedia_labels_eps1.npy')

    posterior_probs = compute_posteriors(commedia_ll, prior_prob)
    commedia_predictions = compute_optimal_Bayes(posterior_probs, C)
    conf_matrix = compute_confusion_matrix(commedia_predictions, commedia_labels)
    print('Confusion matrix:')
    print(conf_matrix)
    emp_bayes_risk = compute_empirical_bayes_risk(conf_matrix, numpy.exp(prior_prob), C, normalize=False)
    print('Empirical Bayes risk:', round(emp_bayes_risk, 3))
    normalized_bayes_risk = compute_empirical_bayes_risk(conf_matrix, numpy.exp(prior_prob), C)
    print('Normalized empirical Bayes risk:', round(normalized_bayes_risk, 3))




