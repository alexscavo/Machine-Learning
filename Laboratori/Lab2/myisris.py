import numpy
import matplotlib
import matplotlib.pyplot as plt

def mcol(v):    #faccio la trasposta
    return v.reshape((v.size), 1)


def load(fileName): #funzione per leggere il dataset e caricare le strutture dati necessarie

    DList = []  #lista 4x150 contenente i sample contenuti nel dataset senza label
    LabelList = []  #vettore di label lette dal dataset
    hLables = {     #dictionary contenente le classi delle label
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica':2
    }
    with open(fileName) as fin: #apro il file
        for line in fin:    #per ogni riga che leggo del file
            try:
                attributes = line.split(',')[0:-1]  #prendo solo gli attributi
                name = line.split(',')[-1].strip()  #prendo il nome della famiglia del fiore

                label = hLables[name]   #"converto" il nome nel numero corrispondente alla label
                verticalAttributes = mcol(numpy.array([float(i) for i in attributes])) #creo un array numpy con i valori letti e salvati in attributes e poi faccio la trasposta del vettore per ottenere un vettore conlonna

                DList.append(verticalAttributes)    #aggiungo alla lista il sample
                LabelList.append(label) #aggiungo la label alla lista della labels
            except:
                pass

    #la lista DList e' una lista di vettori, ma io voglio un vettore di vettori colonna, per questo motivo
    #uso hstack, per stackare in un vettore tutti gli array colonna, ottenendo cosi' un'effettiva matrice 4x150
    return numpy.hstack(DList), numpy.array(LabelList, dtype=numpy.int32)



def plotHist(D, L):

    #voglio plottare un istogramma per ciascuna feature per ciascuna classe
    D0 = D[:, L == 0]   #lista contenente tutti i vettori colonna per i quali la label corrispondente era = 0
    D1 = D[:, L == 1]
    D2 = D[:, L == 2]

    #ricordiamo che i valori delle features sono contenuti nella matrice sotto forma di vettori colonna
    hFeatures = {   #dizionario contenente la corrispondenza numero - feature
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
    }

    #ora voglio visualizzare per ogni feature e per ciascuna classe il plot, quindi faccio un ciclo che mi permette
    #di ciclare sulle 4 feature indicate nel dizionario
    for feat in range(4):
        plt.figure()
        plt.xlabel(hFeatures[feat])
        #bins = numero di bucket, di colonne
        #density = True serve a normalizzare
        #per ciascuna feature (ciascuna riga) prendo tutte le colonne di quella riga
        plt.hist(D0[feat, :], bins = 10, density = True, alpha = 0.4, label = 'Setosa')
        plt.hist(D1[feat, :], bins = 10, density = True, alpha = 0.4, label = 'Versicolor')
        plt.hist(D2[feat, :], bins = 10, density = True, alpha = 0.4, label = 'Virginica')
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('hist_%d.pdf' % feat)
    plt.show()


def plotScatter(D, L):
    
    D0 = D[:, L == 0]   
    D1 = D[:, L == 1]
    D2 = D[:, L == 2]

    hFeatures = {   
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
    }

    #voglio uno scatterplot delle varie coppie di features -> scarto le coppie aventi la stessa feature
    for feat1 in range(4):
        for feat2 in range(4):

            if feat1 == feat2:
                continue

            plt.figure()
            plt.xlabel(hFeatures[feat1])
            plt.ylabel(hFeatures[feat2])
            plt.scatter(D0[feat1, :], D0[feat2, :], label = 'Setosa')
            plt.scatter(D1[feat1, :], D1[feat2, :], label = 'Versicolor')
            plt.scatter(D2[feat1, :], D2[feat2, :], label = 'Virginica')

            plt.legend()
            plt.tight_layout()
            plt.savefig('scatter_%d_%d.pdf' % (feat1, feat2))
        plt.show()



if __name__ == '__main__':

    #change default font size - comment to use default values
    plt.rc('font', size = 16)
    plt.rc('xtick', labelsize = 16)
    plt.rc('ytick', labelsize = 16)

    #plots
    D, L = load('iris.csv')
    #plotHist(D, L)
    #plotScatter(D, L)

    #statistics

    #avg
    #calcola la media di D per ciascuna colonna (indicato dall'1, se fosse stato 0 calcolava la media per righe)
    mu = D.mean(1).reshape(D.shape[0], 1)   #utilizziamo reshape in quanto il risultato di mean e' un vettore orizzontale ma noi vogliamo un vettore colonna

    print('Mean:')
    print(mu)
    print()    

    #broadcasting
    DC = D - mu

    #covarianza
