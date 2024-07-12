import matplotlib
import matplotlib.pyplot
import numpy
import scipy
import scipy.optimize
import functions
import loadData
import plots
import progetto6


def train_dual_SVM(DTR, LTR, C, K = 1):
    
    ZTR = LTR * 2.0 - 1.0 
    DTR_EXT = numpy.vstack([DTR, numpy.ones((1,DTR.shape[1])) * K])
    H = numpy.dot(DTR_EXT.T, DTR_EXT) * functions.mcol(ZTR) * functions.mrow(ZTR)

    def func(alpha):
        Ha = H @ functions.mcol(alpha)
        loss = 0.5 * (functions.mrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(func, numpy.zeros(DTR_EXT.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)
    

    def primalLoss(w_hat):
        S = (functions.mrow(w_hat) @ DTR_EXT).ravel()
        return 0.5 * numpy.linalg.norm(w_hat)**2 + C * numpy.maximum(0, 1 - ZTR * S).sum()

   
    w_hat = (functions.mrow(alphaStar) * functions.mrow(ZTR) * DTR_EXT).sum(1)
    
    
    w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K 

    primalLoss, dualLoss = primalLoss(w_hat), -func(alphaStar)[0]
    print ('SVM - C %e - K %e - primal loss %e - dual loss %e - duality gap %e' % (C, K, primalLoss, dualLoss, primalLoss - dualLoss))
    
    return w, b


def polyKernel(degree, c):
    
    def kernel_function(D1, D2):
        return (numpy.dot(D1.T, D2) + c) ** degree

    return kernel_function

def rbfKernel(gamma):

    def rbfKernelFunc(D1, D2):
        
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = functions.mcol(D1Norms) + functions.mrow(D2Norms) - 2 * numpy.dot(D1.T, D2)
        return numpy.exp(-gamma * Z)

    return rbfKernelFunc


def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps = 1.0):

    ZTR = LTR * 2.0 - 1.0 
    K = kernelFunc(DTR, DTR) + eps
    H = functions.mcol(ZTR) * functions.mrow(ZTR) * K

    
    def func(alpha):
        Ha = H @ functions.mcol(alpha)
        loss = 0.5 * (functions.mrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(func, numpy.zeros(DTR.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)

    print ('SVM (kernel) - C %e - dual loss %e' % (C, -func(alphaStar)[0]))

   
    def fScore(DTE):
        
        K = kernelFunc(DTR, DTE) + eps
        H = functions.mcol(alphaStar) * functions.mcol(ZTR) * K
        return H.sum(0)

    return fScore


# per la minDCF usare progetto6.min_DCF

if __name__ == '__main__':

    D, L = loadData.load('trainData.txt')      # get the data and labels from the dataset
    (DTR, LTR), (DVAL, LVAL) = functions.split_training_test_dataset(D, L)
    K = 1.0
    min_DCFs = []
    act_DCFs = []

    C = numpy.logspace(-5, 0, 11)

    '''for c in C:
        w, b = train_dual_SVM(DTR, LTR, c, K)
        SVAL = (functions.mrow(w) @ DVAL + b).ravel()
        PVAL = (SVAL > 0) * 1
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        DCF_min = progetto6.compute_minDCF(SVAL, LVAL, 0.1, 1.0, 1.0)
        DCF_act = progetto6.compute_actualDCF(SVAL, LVAL, 0.1, 1.0, 1.0)
        min_DCFs.append(DCF_min)
        act_DCFs.append(DCF_act)
        print('minDCF - pT = 0.1:', round(DCF_min, 4))
        print('actDCF - pT = 0.1:', round(DCF_act, 4))
        print() 

    plots.plot_lab9('SVM',min_DCFs, act_DCFs, C)

    # --- CENTERED DATA ---
    min_DCFs = []
    act_DCFs = []

    print('-'*40)
    print('CENTERED DATA SVM')
    mean, _ = functions.compute_mean_covariance(DTR)
    DTR_centered = DTR - mean
    DVAL_centered = DVAL - mean

    for c in C:
        w, b = train_dual_SVM(DTR_centered, LTR, c, K)
        SVAL = (functions.mrow(w) @ DVAL_centered + b).ravel()
        PVAL = (SVAL > 0) * 1
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        DCF_min = progetto6.compute_minDCF(SVAL, LVAL, 0.1, 1.0, 1.0)
        DCF_act = progetto6.compute_actualDCF(SVAL, LVAL, 0.1, 1.0, 1.0)
        min_DCFs.append(DCF_min)
        act_DCFs.append(DCF_act)
        print('minDCF - pT = 0.1:', round(DCF_min, 4))
        print('actDCF - pT = 0.1:', round(DCF_act, 4))
        print() 

    plots.plot_lab9('SVM with centered Data',min_DCFs, act_DCFs, C)'''


    '''# --- POLYNOMIAL KERNEL ---
    print('-'*40)
    print('POLYNOMIAL KERNEL SVM')
    kernelFunc = polyKernel(2, 1)
    eps = 0.0
    min_DCFs = []
    act_DCFs = []

    for c in C:
        fScore = train_dual_SVM_kernel(DTR, LTR, c, kernelFunc, eps)
        SVAL = fScore(DVAL)
        PVAL = (SVAL > 0) * 1
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        DCF_min = progetto6.compute_minDCF(SVAL, LVAL, 0.1, 1.0, 1.0)
        DCF_act = progetto6.compute_actualDCF(SVAL, LVAL, 0.1, 1.0, 1.0)
        min_DCFs.append(DCF_min)
        act_DCFs.append(DCF_act)
        print('minDCF - pT = 0.1:', round(DCF_min, 4))
        print('actDCF - pT = 0.1:', round(DCF_act, 4))
        print() 

    plots.plot_lab9('Polynomial Kernel SVM',min_DCFs, act_DCFs, C)'''

    # --- RBF Kernel---
    print('-'*40)
    print('RBF kernel')
    
    eps = 1.0
    min_DCFs = []
    act_DCFs = []

    C = numpy.logspace(-3, 2, 11)
    Gamma = [numpy.exp(-4), numpy.exp(-3), numpy.exp(-2), numpy.exp(-1)]

    for gamma in Gamma:
        min_DCFs = []
        act_DCFs = []
        for c in C:
            kernelFunc = rbfKernel(gamma)
            fScore = train_dual_SVM_kernel(DTR, LTR, c, kernelFunc, eps)
            SVAL = fScore(DVAL)
            PVAL = (SVAL > 0) * 1
            err = (PVAL != LVAL).sum() / float(LVAL.size)
            DCF_min = progetto6.compute_minDCF(SVAL, LVAL, 0.1, 1.0, 1.0)
            DCF_act = progetto6.compute_actualDCF(SVAL, LVAL, 0.1, 1.0, 1.0)
            min_DCFs.append(DCF_min)
            act_DCFs.append(DCF_act)
            print('minDCF - pT = 0.1:', round(DCF_min, 4))
            print('actDCF - pT = 0.1:', round(DCF_act, 4))
            print() 

        title = 'RBF Kernel SVM: C=%e Gamma=%e' % c, gamma

        plots.plot_lab9(title ,min_DCFs, act_DCFs, C)

    '''# --- OPTIONAL ---
    print('-'*40)
    print('OPTIONAL')
    kernelFunc = polyKernel(4, 1)
    eps = 0.0
    min_DCFs = []
    act_DCFs = []

    for c in C:
        fScore = train_dual_SVM_kernel(DTR, LTR, c, kernelFunc, eps)
        SVAL = fScore(DVAL)
        PVAL = (SVAL > 0) * 1
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        DCF_min = progetto6.compute_minDCF(SVAL, LVAL, 0.1, 1.0, 1.0)
        DCF_act = progetto6.compute_actualDCF(SVAL, LVAL, 0.1, 1.0, 1.0)
        min_DCFs.append(DCF_min)
        act_DCFs.append(DCF_act)
        print('minDCF - pT = 0.1:', round(DCF_min, 4))
        print('actDCF - pT = 0.1:', round(DCF_act, 4))
        print() 

    plots.plot_lab9('Polynomial Kernel SVM with degree = 4',min_DCFs, act_DCFs, C)'''
