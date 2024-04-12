import numpy as np
import matplotlib
import matplotlib.pyplot as plt



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


def compute_mean(D, L, class_number):

    D_class = D[:, L==class_number]

    mu_c = np.mean(D_class, axis=0)
    
    return mu_c
    


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

    print("/---------------------------------/\n")

    mu_c0 = compute_mean(D, L, 0)

    mu_c1 = compute_mean(D, L, 1)

    mu_c2 = compute_mean(D, L, 2)

    print("mu_c0 = ", mu_c0)
    print("mu_c1 = ", mu_c1)
    print("mu_c2 = ", mu_c2)

    vett_mu_class = [mu_c0, mu_c1, mu_c2]
    sum = 0

    print(vett_mu_class)

    for i in [0, 2]:
        mu_class = vett_mu_class[i]

        diff_mu = mu_class - mu

        diff_mu_t = diff_mu.T

        sum += np.matmul(diff_mu, diff_mu_t) * 3

    S_B = sum / N

    print("S_B = ", S_B)

    



