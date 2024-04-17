import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg as scipy


def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))

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

    s, U = scipy.eigh(SB, SW)
    W = U[:, ::-1][:, 0:2]

    UW, _, _ = np.linalg.svd(W)
    U = UW[:, 0:2]

    print("W = ", W)
    print("U = ", U)