import numpy
import load


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

    print(words_per_class)
    
    for cls in tercets_train:
        words_num = sum(words_per_class[cls].values())  # conto il numero di occorrenze totali di tutte le parole del canto
        
        for word in words_per_class[cls]:
            words_per_class[cls][word] = numpy.log(words_per_class[cls][word]) - numpy.log(words_num)   # log(N_{c,j} / N_c)




if __name__ == '__main__':

    # carico il dataset
    lInf, lPur, lPar = load.load_data()

    # splitto i dataset per ottenere training e validation (25% validation)
    lInf_train, lInf_evaluation = load.split_data(lInf, 4)
    lPur_train, lPur_evaluation = load.split_data(lPur, 4)
    lPar_train, lPar_evaluation = load.split_data(lPar, 4)

    # ----- SOLUZIONE 1 -----

    class_to_index = {'inferno': 0, 'purgatorio': 1,  'paradiso': 2}    # serve per il mapping delle parole e con gli indici

    tercets_train = {       # dizionario con chiave la classe e come valore la lista di terzine di quella classe
        'inferno': lInf_train,  
        'purgatorio': lPur_train,
        'paradiso': lPar_train
    }

    tercets_eval = lInf_evaluation + lPar_evaluation + lPur_evaluation  # l'evaluation lo faccio su tutti gli evaluation set

    sol1_model = sol1_estimate_model(tercets_train, eps = 0.001)


