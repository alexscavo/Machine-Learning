import numpy
import scipy
import scipy.special
import load

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))


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


def compute_accuracy(posterior_prob, labels):

    predicted_labels = numpy.argmax(posterior_prob, axis = 0)
    #predicted_labels = posterior_prob.argmax(0)  # faccio la predizione in base alla posterior prob
    NCorrect = (predicted_labels.ravel() == labels.ravel()).sum()   # numero di predictions corrette
    Ntotal = labels.size
    accuracy = float(NCorrect * 100) / float(Ntotal) 

    return accuracy 

if __name__ == '__main__':

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

    labelsInf = numpy.zeros(len(lInf_evaluation)) # inizializzo il vettore perchè non potrei altrimenti assegnarvi valori
    labelsInf[:] = class_to_index['inferno']    # assegno a tutti i valori del vettore la classe equivalente ad 'inferno'

    labelsPur = numpy.zeros(len(lPur_evaluation)) # inizializzo il vettore perchè non potrei altrimenti assegnarvi valori
    labelsPur[:] = class_to_index['purgatorio']    # assegno a tutti i valori del vettore la classe equivalente ad 'inferno'

    labelsPar = numpy.zeros(len(lPar_evaluation)) # inizializzo il vettore perchè non potrei altrimenti assegnarvi valori
    labelsPar[:] = class_to_index['paradiso']    # assegno a tutti i valori del vettore la classe equivalente ad 'inferno'

    labels_eval = numpy.hstack([labelsInf, labelsPur, labelsPar])    # impilo i 3 vettori di labels

    print('----- accuracies for each class: -----\n')
    print('Inferno: %.2f%%' % (compute_accuracy(posterior_prob[:, labels_eval==class_to_index['inferno']], labels_eval[labels_eval==class_to_index['inferno']])))
    print('Purgatorio: %.2f%%' % (compute_accuracy(posterior_prob[:, labels_eval==class_to_index['purgatorio']], labels_eval[labels_eval==class_to_index['purgatorio']])))
    print('Paradiso: %.2f%%' % (compute_accuracy(posterior_prob[:, labels_eval==class_to_index['paradiso']], labels_eval[labels_eval==class_to_index['paradiso']])))

    print('----- Total model accuracy -----\n')
    print('Model accuracy: %.2f%%' % (compute_accuracy(posterior_prob, labels_eval)))
