import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg as scipy
import sklearn.datasets


def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))

def load_iris():
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def load(fname):
    DList = []
    labelsList = []
    hLabels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
        }

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = mcol(np.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return np.hstack(DList), np.array(labelsList, dtype=np.int32)

def plot_hist(D, L):

    D1 = D[L==1]
    D2 = D[L==2]

    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
        }


    plt.figure()
    plt.hist(D1, bins = 5, density = True, alpha = 0.4, label = 'Versicolor')
    plt.hist(D2, bins = 5, density = True, alpha = 0.4, label = 'Virginica')
    
    plt.legend()
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.savefig('hist_%d.pdf' % 5)
    plt.show()

def plot_scatter(D, L):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]

    plt.figure()
    plt.scatter(D0[0, :], D0[1, :], label = 'Setosa')
    plt.scatter(D1[0, :], D1[1, :], label = 'Versicolor')
    plt.scatter(D2[0, :], D2[1, :], label = 'Virginica')

    plt.legend()
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.savefig('scatter_%d_%d.pdf' % (0, 1))
    plt.show()


def split_training_test_dataset(D, L, seed = 0):    #funzione per splittare il dataset creando un test set e un training set

    nTrain = int(D.shape[1]*2.0/3.0)    #numero di sample usati per il training
    
    np.random.seed(seed)    #genero seed casuale per le permutazioni

    idx = np.random.permutation(D.shape[1]) #genero vettore di numeri tra 0 e 150 disposti randomicamente
    idxTrain = idx[0:nTrain]    #seleziono per il training solo i primi nTrain numeri disposti randomicamente
    idxTest = idx[nTrain:]      #seleziono per il test solo gli ultimi numeri rimasti

    DTR = D[:, idxTrain]    #vado a fare un sampling del dataset D selezionando i sample nelle posizioni corrispondenti a idxTrain, creando cosi' i due set di test e training
    DVAL = D[:, idxTest]    #stessa cosa ma per il test set
    
    LTR = L[idxTrain]   #stessa cosa con ma per le label
    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)




#------MAIN-----#
if __name__ == '__main__':

    D, L = load('iris.csv')
    N = D.shape[1]

    mu = D.mean(1)  #calcolo la media delle colonne (somma di tutti i vettori colonna / numero di colonne)  
                    #sto facendo la media di ciascuna feature, ottenendo quindi 1 solo vettore colonna

    print(mu)
    DC = D - mcol(mu)   #ottengo la matrice centrata, rimuovendo da ciascun sample la media, come se stessi normalizzando

    C = (1/N)*(DC @ DC.T)   #calcolo la matrice delle covarianze

    print(C)

    s, U = np.linalg.eigh(C)    #calcolo gli autovalori, ottenendoli ordinati dal più piccolo al più grande e in U i corrispondenti autovettori


    for m in [2, 3]:
        P = U[:, ::-1][:, 0:m]  #noi vogliamo però U ordinata in modo decrescente, quindi dobbiamo invertire la posizione delle colonne

        #CALCOLO SVD
        # s = Sigma = matrice contenente sulla diagonale i singular values di C, ordinati in modo decrescente
        # U = U = le colonne di U sono gli autovettori corrispondenti ai singular values, ovvero i left singual vectors
        # vh = V^T = contiene i right singual values 
        U, s, Vh = np.linalg.svd(C)  #calcolo la svd sulla matrice C     

        DP = np.dot(P.T, D)   # calcolo le proiezioni di tutti i punti


        print("autoevttori con m = ", m)
        print(U)

        #if m == 2:
            #plot_scatter(DP, L)


    data = np.load('IRIS_PCA_matrix_m4.npy')

    print("autovettori ordinati\n", data)

    print("/--------------LDA----------------/\n")

    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]

    mu0 = D0.mean(1)
    D0C = D0 - mcol(mu0)
    SW0 = D0C @ D0C.T

    mu1 = D1.mean(1)
    D1C = D1 - mcol(mu1)
    SW1 = D1C @ D1C.T

    mu2 = D2.mean(1)
    D2C = D2 - mcol(mu2)
    SW2 = D2C @ D2C.T

    SW = (1/D.shape[1]) * (SW0 + SW1 + SW2)

    print("SW = ", SW)


    mu0 = mcol(mu0)
    mu1 = mcol(mu1)
    mu2 = mcol(mu2)
    mu = mcol(mu)

    SB0 = D0.shape[1] * ((mu0 - mu) @ (mu0 - mu).T)
    SB1 = D1.shape[1] * ((mu1- mu) @ (mu1 - mu).T)
    SB2 = D2.shape[1] * ((mu2 - mu) @ (mu2 - mu).T)

    SB = (1/N) * (SB0 + SB1 + SB2)

    print("SB = ", SB)

    '''Generalized eigenvalue problem'''

    s, U = scipy.eigh(SB, SW)
    W1 = U[:, ::-1][:, 0:2]


    print("W1 = ", W1)


    '''Joint diagonalization os SB and SW to solve the eigenvalue problem'''

    U, s, _ = np.linalg.svd(SW)

    P1 = np.dot( np.dot(U, np.diag(1.0/(s**0.5))), U.T)

    SBT = P1 @ SB @ P1.T    #contiene gli autovettori piu' grandi

    print("SBT = ")
    print(SBT)

    P2, s, _ = np.linalg.svd(SBT)

    P2 = P2[:, 0:2]

    print("P1.T:\n", P1.T, "\nP2:\n", P2)

    W2 = P1.T @ P2

    print("W2:\n", W2)

    solution = np.load('IRIS_LDA_matrix_m2.npy')

    print("soluzione:\n", solution)

    print(np.linalg.svd(np.hstack([W1, solution]))[1])

    print(np.linalg.svd(np.hstack([W2, solution]))[1])

    
    LDA_DATASET = W1.T @ D

    #plot_scatter(LDA_DATASET, L)

    '''Quindi in pratica posso calcolare la matrice della LDA (ovvero W) in uno dei due modi visti, ovvero
        -   Tramite generalized eigenvalue problem
        -   Tramite joint diagonalization per risolvere l'eigenvalue problem'''
    

    '''---------PCA + LDA-------'''

    DIris, LIris = load_iris()
    D = DIris[:, LIris != 0]    #contiene i samples delle classi 1 e 2
    L = LIris[LIris != 0]   #contiene le labels 1 e 2


    (DTR, LTR), (DVAL, LVAL) = split_training_test_dataset(D, L)

    DTR1 = DTR[:, LTR == 1]
    DTR2 = DTR[:, LTR == 2]

    mu_tr1 = DTR1.mean(1)
    mu_tr2 = DTR2.mean(1)
    mu_tr = DTR.mean(1)

    DTR1C = DTR1 - mcol(mu_tr1)
    DTR2C = DTR2 - mcol(mu_tr2)

    SWTR1 = DTR1C @ DTR1C.T
    SWTR2 = DTR2C @ DTR2C.T

    SWTR = (1/DTR.shape[1]) * (SWTR1 + SWTR2)

    print("SWTR =\n", SWTR)

    mu_tr1 = mcol(mu_tr1)
    mu_tr2 = mcol(mu_tr2)
    mu_tr = mcol(mu_tr)

    SBTR1 = DTR1.shape[1] * ((mu_tr1- mu_tr) @ (mu_tr1 - mu_tr).T)
    SBTR2 = DTR2.shape[1] * ((mu_tr2 - mu_tr) @ (mu_tr2 - mu_tr).T)

    SBTR = (1/N) * (SBTR1 + SBTR2)

    print("SBTR =\n", SBTR)

    s_tr, UTR = scipy.eigh(SBTR, SWTR)

    WTR = UTR[:, ::-1][:, 0]    #modello allenato

    print("WTR =\n", WTR)

    LDA_result = WTR.T @ DVAL   
    LDA_training = WTR.T @ DTR

    plot_hist(LDA_training, LTR)
    plot_hist(LDA_result, LVAL)


