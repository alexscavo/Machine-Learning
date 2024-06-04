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


def compute_llr(D, parameters):
    pdf_1 = logpdf_GAU_ND(D, parameters[1][0], parameters[1][1])
    pdf_0 = logpdf_GAU_ND(D, parameters[0][0], parameters[0][1])

    return pdf_1 - pdf_0 


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


def compute_parameters_naive_bayes(D, L):
    labels = set(L)
    parameters = {}
    for label in labels:
        DX = D[:, L== label]
        mu, C = compute_mean_covariance(DX)
        C = C * numpy.identity(C.shape[0])  # remove every off-diagonal element multiplying the covariance matrix with the identity matrix
        parameters[label] = (mu, C)

    return parameters