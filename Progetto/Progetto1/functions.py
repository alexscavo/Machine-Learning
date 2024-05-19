import numpy

def mcol(v):    #transposed vector vertically
    return v.reshape((v.size, 1))

def mrow(v):    #transposed vector horitzonally
    return v.reshape((1, v.size))

def split_training_test_dataset(D, L, seed = 0):    #funzione per splittare il dataset creando un test set e un training set

    nTrain = int(D.shape[1]*2.0/3.0)    #numero di sample usati per il training
    
    numpy.random.seed(seed)    #genero seed casuale per le permutazioni

    idx = numpy.random.permutation(D.shape[1]) #genero vettore di numeri tra 0 e 150 disposti randomicamente
    idxTrain = idx[0:nTrain]    #seleziono per il training solo i primi nTrain numeri disposti randomicamente
    idxTest = idx[nTrain:]      #seleziono per il test solo gli ultimi numeri rimasti

    DTR = D[:, idxTrain]    #vado a fare un sampling del dataset D selezionando i sample nelle posizioni corrispondenti a idxTrain, creando cosi' i due set di test e training
    DVAL = D[:, idxTest]    #stessa cosa ma per il test set
    
    LTR = L[idxTrain]   #stessa cosa con ma per le label
    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)

def compute_mean_covariance(D):
    N = float(D.shape[1])
    mu = mcol(D.mean(1))
    DC = D - mu
    C = (DC @ DC.T) / N

    return mu, C

def print_matrix(M):

    for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                print(M[i][j], end =", "),
            print()

def PCA_matrix(D, m):

    N = D.shape[1]      # total number of samples
    mu = D.mean(1)  # dataset mean (columns mean)

    DC = D - mcol(mu)     #centered dataset

    C = float(1/N) * (DC @ DC.T)     # covariance matrix

    U, s, Vh = numpy.linalg.svd(C)      # svd on covariance matrix

    P = U[:, 0:m]

    return P