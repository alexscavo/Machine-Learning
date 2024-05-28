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

    conf_mat = numpy.zeros((len(set(L)), len(set(L))), dtype = int)

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
    threshold = -numpy.log((prior * Cfn)/((1-prior)*Cfp))   # uso la formula per calcolare la threshold e seleziono poi i llr > threshold

    return numpy.int32(llr > threshold)







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


    # carico il dataset
    lInf, lPur, lPar = load.load_data()

    # splitto i dataset per ottenere training e validation (25% validation)
    lInf_train, lInf_evaluation = load.split_data(lInf, 4)
    lPur_train, lPur_evaluation = load.split_data(lPur, 4)
    lPar_train, lPar_evaluation = load.split_data(lPar, 4)

    # prior probabilities
    prior_prob = numpy.log(mcol(numpy.ones(3)/.3))

    # ----- SOLUZIONE 1 -----

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

    for prior, Cfn, Cfp in [(0.5, 1, 1), (0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]:  #
        print()
        print('Prior:', prior, '- Cfn:', Cfn, '- Cfp:', Cfp)

        predictions_binary = compute_optimal_bayes_binary_llr(llr_commedia, prior, Cfn, Cfp)
        conf_matrix = compute_confusion_matrix(predictions_binary, labels_commedia)
        print(conf_matrix)