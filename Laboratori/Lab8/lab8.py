import matplotlib
import matplotlib.pyplot
import numpy
import scipy
import scipy.optimize
import sklearn.datasets

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))


def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]
    L = L[L != 0]
    L[L==2] = 0

    return D, L

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

def f(x):   # x ha 2 righe: la prima contiene la y, la seconda la z

    y = x[0]    
    z = x[1]

    f = (y + 3)**2 + numpy.sin(y) + (z + 1)**2

    return f

def fprime(x):   # x ha 2 righe: la prima contiene la y, la seconda la z

    y = x[0]    
    z = x[1]

    f = numpy.array([2*(y + 3) + numpy.cos(y), 2*(z+1)])

    return f


def trainLogReg(DTR, LTR, l):

    ZTR = LTR * 2.0 - 1     # sarebbe la z che moltiplica s nella sommatoria

    def logreg_obj(v):  # contiene i parametri per la logistic regression ==> v = [w, b]

        w, b = v[0: -1], v[-1]  # spacchetto i parametri    
        s = (w.T @ DTR).ravel() + b     # s (separation surface)

        loss = numpy.logaddexp(0, -ZTR * s)      

        reg_term = (l/2)*(numpy.linalg.norm(w)**2)

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        GW = (mrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()

        objective_function = reg_term + loss.mean()

        print(objective_function)

        return objective_function, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(func = logreg_obj, x0 = numpy.zeros(DTR.shape[0]+1))[0] # cerco il minimo della funzione obbiettivo









########################################################
#                                                      #
#-------------------------MAIN-------------------------#
#                                                      #
########################################################
if __name__ == '__main__':

    D, L = load_iris_binary()      # carico il dataset
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)      # splitto il dataset in training e evaluation

    #print(scipy.optimize.fmin_l_bfgs_b(func = f, approx_grad = True, x0 = numpy.zeros(2)))
    #print(scipy.optimize.fmin_l_bfgs_b(func = f, fprime = fprime, x0 = numpy.zeros(2)))


    for _lambda in [1e-3, 1e-1, 1.0]:
        trainLogReg(DTR, LTR, _lambda)