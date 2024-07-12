import numpy
import functions
import scipy
import loadData
import progetto6
import matplotlib.pyplot as plt
import plots
import progetto7


def plot_comparison(results):
    plt.figure(figsize=(12, 8))

    for model in results:
        min_DCF = results[model]["min_DCF"]
        act_DCF = results[model]["act_DCF"]
        x = results[model]["x"]

        plt.plot(x, min_DCF, label=f'{model} Minimum DCF', marker='o')
        plt.plot(x, act_DCF, label=f'{model} Actual DCF', marker='x')

    plt.xscale('log', base=10)
    plt.xlabel('Parameter (C for LR and SVM, Components for GMM)')
    plt.ylabel('DCF')
    plt.title('DCF Comparison for LR, SVM, and GMM')
    plt.legend()
    plt.grid(True)
    plt.show()




def logpdf_GMM(X, gmm):

    S = []
    
    for w, mu, C in gmm:  # con gmm calcolo i parametri
        logpdf_conditional = functions.logpdf_GAU_ND(X, mu, C)     # calcolo la conditional probability
        logpdf_joint = logpdf_conditional + numpy.log(w)  # calcolo la joint probability
        S.append(logpdf_joint)      # aggiungo alla matrice S
        
    S = numpy.vstack(S)
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens

def smooth_covariance_matrix(C, psi):

    U, s, Vh = numpy.linalg.svd(C)
    s[s<psi]=psi
    CUpd = U @ (functions.mcol(s) * U.T)
    return CUpd

def split_GMM_LBG(gmm, alpha = 0.1, verbose=True):

    gmmOut = []
    if verbose:
        print ('LBG - going from %d to %d components' % (len(gmm), len(gmm)*2))
    for (w, mu, C) in gmm:
        U, s, Vh = numpy.linalg.svd(C)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        gmmOut.append((0.5 * w, mu - d, C))
        gmmOut.append((0.5 * w, mu + d, C))
    return gmmOut

def train_GMM_EM_Iteration(X, gmm, covType = 'Full', psiEig = None): 

    assert (covType.lower() in ['full', 'diagonal', 'tied'])
    
    # E-step
    S = []
    
    for w, mu, C in gmm:
        logpdf_conditional = functions.logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + numpy.log(w)
        S.append(logpdf_joint)
        
    S = numpy.vstack(S) # Compute joint densities f(x_i, c), i=1...n, c=1...G
    logdens = scipy.special.logsumexp(S, axis=0) # Compute marginal for samples f(x_i)

    # Compute posterior for all clusters - log P(C=c|X=x_i) = log f(x_i, c) - log f(x_i)) - i=1...n, c=1...G
    # Each row for gammaAllComponents corresponds to a Gaussian component
    # Each column corresponds to a sample (similar to the matrix of class posterior probabilities in Lab 5, but here the rows are associated to clusters rather than to classes
    gammaAllComponents = numpy.exp(S - logdens)

    # M-step
    gmmUpd = []
    for gIdx in range(len(gmm)): 
    # Compute statistics:
        gamma = gammaAllComponents[gIdx] # Extract the responsibilities for component gIdx
        Z = gamma.sum()
        F = functions.mcol((functions.mrow(gamma) * X).sum(1)) # Exploit broadcasting to compute the sum
        S = (functions.mrow(gamma) * X) @ X.T
        muUpd = F/Z
        CUpd = S/Z - muUpd @ muUpd.T
        wUpd = Z / X.shape[1]
        if covType.lower() == 'diagonal':
            CUpd  = CUpd * numpy.eye(X.shape[0]) # An efficient implementation would store and employ only the diagonal terms, but is out of the scope of this script
        gmmUpd.append((wUpd, muUpd, CUpd))

    if covType.lower() == 'tied':
        CTied = 0
        for w, mu, C in gmmUpd:
            CTied += w * C
        gmmUpd = [(w, mu, CTied) for w, mu, C in gmmUpd]

    if psiEig is not None:
        gmmUpd = [(w, mu, smooth_covariance_matrix(C, psiEig)) for w, mu, C in gmmUpd]
        
    return gmmUpd

def train_GMM_EM(X, gmm, covType = 'Full', psiEig = None, epsLLAverage = 1e-6, verbose=True):

    llOld = logpdf_GMM(X, gmm).mean()
    llDelta = None
    if verbose:
        print('GMM - it %3d - average ll %.8e' % (0, llOld))
    it = 1
    while (llDelta is None or llDelta > epsLLAverage):
        gmmUpd = train_GMM_EM_Iteration(X, gmm, covType = covType, psiEig = psiEig)
        llUpd = logpdf_GMM(X, gmmUpd).mean()
        llDelta = llUpd - llOld
        if verbose:
            print('GMM - it %3d - average ll %.8e' % (it, llUpd))
        gmm = gmmUpd
        llOld = llUpd
        it = it + 1

    if verbose:
        print('GMM - it %3d - average ll %.8e (eps = %e)' % (it, llUpd, epsLLAverage))        
    return gmm


def train_GMM_LBG_EM(X, numComponents, covType = 'Full', psiEig = None, epsLLAverage = 1e-6, lbgAlpha = 0.1, verbose=True):

    mu, C = functions.compute_mean_covariance(X)

    if covType.lower() == 'diagonal':
        C = C * numpy.eye(X.shape[0]) # We need an initial diagonal GMM to train a diagonal GMM
    
    if psiEig is not None:
        gmm = [(1.0, mu, smooth_covariance_matrix(C, psiEig))] # 1-component model - if we impose the eignevalus constraint, we must do it for the initial 1-component GMM as well
    else:
        gmm = [(1.0, mu, C)] # 1-component model
    
    while len(gmm) < numComponents:
        # Split the components
        if verbose:
            print ('Average ll before LBG: %.8e' % logpdf_GMM(X, gmm).mean())
        gmm = split_GMM_LBG(gmm, lbgAlpha, verbose=verbose)
        if verbose:
            print ('Average ll after LBG: %.8e' % logpdf_GMM(X, gmm).mean()) # NOTE: just after LBG the ll CAN be lower than before the LBG - LBG does not optimize the ll, it just increases the number of components
        # Run the EM for the new GMM
        gmm = train_GMM_EM(X, gmm, covType = covType, psiEig = psiEig, verbose=verbose, epsLLAverage = epsLLAverage)
    return gmm




if __name__ == '__main__':

    D, L = loadData.load('trainData.txt')      # get the data and labels from the dataset
    (DTR, LTR), (DVAL, LVAL) = functions.split_training_test_dataset(D, L)

    min_DCFs = []
    act_DCFs = []

    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]

    components = [1,2,4,8,16,32]
    results = {}
    best_min_DCF = 100
    best_parameters = []

    # --- FULL COVARIANCE MATRIX ---
    covType = 'full'
    print('-'*40)
    print(covType)
    for numComponents0 in components:
        min_DCFs = []
        act_DCFs = []
        for numComponents1 in components:

            print('False class components: ', int(numComponents0), ' - True class components: ', int(numComponents1))
            gmm0 = train_GMM_LBG_EM(D0, int(numComponents0), covType = covType, verbose=False, psiEig = 0.01)
            gmm1 = train_GMM_LBG_EM(D1, int(numComponents1), covType = covType, verbose=False, psiEig = 0.01)

            SLLR = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)

            DCF_min = progetto6.compute_minDCF(SLLR, LVAL, 0.1, 1.0, 1.0)
            DCF_act = progetto6.compute_actualDCF(SLLR, LVAL, 0.1, 1.0, 1.0)
            min_DCFs.append(DCF_min)
            act_DCFs.append(DCF_act)

            if(best_min_DCF > DCF_min):
                best_min_DCF = DCF_min
                best_parameters = [numComponents0, numComponents1]

            print('minDCF - pT = 0.1:', round(DCF_min,4))
            print('actDCF - pT = 0.1:', round(DCF_act,4))
        print('---')

        plt.figure()
        plt.plot(components, min_DCFs, label="min DCF", color="b", marker='o')
        plt.plot(components, act_DCFs, label="actual DCF", color="r", marker='x')
        plt.xlabel("True class number of components")
        plt.ylabel("DCF value")
        plt.title("False class number of components fixed to: %d" %(int(numComponents0)))
        plt.legend()
        plt.grid(True)
        #plt.savefig("plots_p8/standard_gmm_false_component_%d.pdf"%(int(numComponents0))) 
        #plt.show()

    '''# --- FULL COVARIANCE MATRIX ---
    covType = 'diagonal'
    print('-'*40)
    print(covType)
    for numComponents0 in components:
        min_DCFs = []
        act_DCFs = []
        for numComponents1 in components:

            print('False class components: ', int(numComponents0), ' - True class components: ', int(numComponents1))
            gmm0 = train_GMM_LBG_EM(D0, int(numComponents0), covType = covType, verbose=False, psiEig = 0.01)
            gmm1 = train_GMM_LBG_EM(D1, int(numComponents1), covType = covType, verbose=False, psiEig = 0.01)

            SLLR = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)

            DCF_min = progetto6.compute_minDCF(SLLR, LVAL, 0.1, 1.0, 1.0)
            DCF_act = progetto6.compute_actualDCF(SLLR, LVAL, 0.1, 1.0, 1.0)
            min_DCFs.append(DCF_min)
            act_DCFs.append(DCF_act)
            print('minDCF - pT = 0.1:', round(DCF_min,4))
            print('actDCF - pT = 0.1:', round(DCF_act,4))
        print('---')'''


    # --- MODELS COMPARISON --- 
    components = [1, 2, 3, 4, 5]
    C = [0.1, 1, 10]
    lambda_values = numpy.logspace(-4, 2, 13)

    # Storing results for plotting
    results = {
        "LR": {"min_DCF": [], "act_DCF": [], "x": []},
        "SVM": {"min_DCF": [], "act_DCF": [], "x": []},
        "GMM": {"min_DCF": [], "act_DCF": [], "x": []}
    }

    DTR_expanded = progetto6.quadratic_feature_expansion(DTR)
    DVAL_expanded = progetto6.quadratic_feature_expansion(DVAL)

    #logistic regression
    w, b = progetto6.trainLogReg(DTR_expanded, LTR, 3.16227766e-04)
    
    #SVM rbf kernel
    kernelFunc = progetto7.rbfKernel(numpy.exp(-4))
    fScore = progetto7.train_dual_SVM_kernel(DTR, LTR, _lambda, kernelFunc, 1.0)



    # Logistic Regression
    for _lambda in lambda_values:
        w, b = progetto6.trainLogReg(DTR_expanded, LTR, _lambda)   # calcolo i parametri del modello, w e b
        Sval = w.T @ DVAL_expanded + b

        emp_prior = (LTR == 1).sum() / float(LTR.size)
        Sllr = Sval - numpy.log(emp_prior / (1-emp_prior))     
        DCF_min = progetto6.compute_minDCF(Sllr, LVAL, 0.1, 1.0, 1.0)
        DCF_act = progetto6.compute_actualDCF(Sllr, LVAL, 0.1, 1.0, 1.0)
        min_DCFs.append(DCF_min)
        act_DCFs.append(DCF_act)
        results["LR"]["min_DCF"].append(DCF_min)
        results["LR"]["act_DCF"].append(DCF_act)
        results["LR"]["x"].append(_lambda)

    # SVM
    eps = 1.0
    for _lambda in lambda_values:
        kernelFunc = progetto7.rbfKernel(numpy.exp(-4))
        fScore = progetto7.train_dual_SVM_kernel(DTR, LTR, _lambda, kernelFunc, eps)
        SVAL = fScore(DVAL)
        PVAL = (SVAL > 0) * 1
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        DCF_min = progetto6.compute_minDCF(SVAL, LVAL, 0.1, 1.0, 1.0)
        DCF_act = progetto6.compute_actualDCF(SVAL, LVAL, 0.1, 1.0, 1.0)
        min_DCFs.append(DCF_min)
        act_DCFs.append(DCF_act)
        results["SVM"]["min_DCF"].append(DCF_min)
        results["SVM"]["act_DCF"].append(DCF_act)
        results["SVM"]["x"].append(_lambda)

    # GMM
    covType = 'full'
    for _lambda in lambda_values:
        gmm0 = train_GMM_LBG_EM(DTR, numComponents0, covType=covType, verbose=False, psiEig=0.01)
        gmm1 = train_GMM_LBG_EM(DTR, numComponents1, covType=covType, verbose=False, psiEig=0.01)
        SLLR = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
        DCF_min = progetto6.compute_minDCF(SLLR, LVAL, 0.1, 1.0, 1.0)
        DCF_act = progetto6.compute_actualDCF(SLLR, LVAL, 0.1, 1.0, 1.0)
        results["GMM"]["min_DCF"].append(DCF_min)
        results["GMM"]["act_DCF"].append(DCF_act)
        results["GMM"]["x"].append(_lambda)

    plot_comparison(results)

