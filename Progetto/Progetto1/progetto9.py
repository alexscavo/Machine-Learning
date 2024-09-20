import numpy
import scipy.optimize
import matplotlib.pyplot as plt
import gmm
import progetto5
import progetto6
import functions
import loadData
import progetto7
import progetto8



def DCF_normalized(prior, Cfn, Cfp, llrs, LTE):
    """
    We now compute the normalized DCF (valid only for *binary* tasks) as prior*Cfn*Pfn + (1-prior)*Cfp*Pfp,
    where Pfn is the false negative rate and Pfp is the false positive rate.
    The Bayes risk (DCF un-normalized) allows us comparing different systems, however it does not tell us what is the benefit of
    using our recognizer with respect to optimal decisions based on *prior* information only. We can compute
    a normalized detection cost, by dividing the Bayes risk by the risk of an optimal system that does not
    use the test data at all, which means that is based only on the triplet (prior, Cfn, Cfp).
    The Bayes risk of such a system is therefore the minimum bewteen prior*Cfn and (1-prior)*Cfp, which means that
    it assigns a label based only on costs (given a priory by the application) and prior probailities fo each class

    """
    predictions = progetto5.compute_optimal_bayes_binary_llr(llrs, prior, Cfn, Cfp)
    conf_matrix = progetto5.compute_confusion_matrix(predictions, LTE)

    bRisk_dummy = min(prior*Cfn, (1-prior)*Cfp)

    Pfn = conf_matrix[0,1]/(conf_matrix[0,1]+conf_matrix[1,1])
    Pfp = conf_matrix[1,0]/(conf_matrix[1,0]+conf_matrix[0,0])
    bRisk = prior*Cfn*Pfn + (1-prior)*Cfp*Pfp

    bRisk_normalized = bRisk/bRisk_dummy

    return bRisk_normalized

def min_DCF(prior, Cfn, Cfp, llrs, LTE):
    """
    Normalized minimum detection cost function is computed here.

    To compute the minimum cost, consider a set of thresholds corresponding to (-inf, s1, ..., sM, +inf), where
    sM are the test scores, sorted in increasing order (notice that the DCF can change only when we
    change a prediction, and that can happen only when the threshold moves “across” one of the evaluation
    scores). For each threshold t, compute the confusion matrix on the test set itself that would be obtained
    if scores were thresholded at t, and the corresponding normalized DCF using the code developed in the
    previous section. In practice the minimum threashold that we used here is s1-0.1, because in this case we evaluate
    all the samples as belonging to class 1. The maximum threshold is sM, because in this case we evaluate
    all the samples as belonging to class 0
    """
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), numpy.sort(llrs), numpy.array([numpy.inf])])    
    bRisk_dummy = min(prior*Cfn, (1-prior)*Cfp)
    min_risk = 0
    for i in range(thresholds.shape[0]):
        predictions = numpy.full(llrs.shape, 0)
        t = thresholds[i] #t is the threshold
        predictions[llrs > t] = 1
        predictions[llrs <= t] = 0
        conf_matrix = progetto5.compute_confusion_matrix(predictions, LTE)
        Pfn = conf_matrix[0,1]/(conf_matrix[0,1]+conf_matrix[1,1])
        Pfp = conf_matrix[1,0]/(conf_matrix[1,0]+conf_matrix[0,0])
        bRisk = prior*Cfn*Pfn + (1-prior)*Cfp*Pfp
        bRisk_normalized = bRisk/bRisk_dummy
        if(i == 0 or bRisk_normalized < min_risk):
            min_risk = bRisk_normalized
    return min_risk

def bayes_error_plots(llrs, labels, tag, colors):
    """ 
    Bayes error plot

    This function computes the bayes error plot given the llrs and the ACTUAL labels of the samples.
    "tag" parameter represent the name of the classifier you want to show on the legend
    "colors" parameter is has to be a 2-elements vector, each element is the color preferred 
    for actual DCF e min DCF respectively 
    """

    effPriorLogOdds = numpy.linspace(-3, 3, 21)

    effPrior = 1/(1+numpy.exp(-effPriorLogOdds))

    Cfn = 1
    Cfp = 1
    dcf = numpy.zeros(effPrior.shape)
    mindcf = numpy.zeros(effPrior.shape)
    for i in range(effPrior.shape[0]):
        pi = effPrior[i]
        dcf[i] = DCF_normalized(pi, Cfn,  Cfp, llrs, labels)
        mindcf[i] = min_DCF(pi, Cfn, Cfp, llrs, labels)


    plt.plot(effPriorLogOdds, dcf, label=tag+" actual DCF", color=colors[0])
    plt.plot(effPriorLogOdds, mindcf, label=tag+ " min DCF", color=colors[1], linestyle='--')
    plt.ylim([0, 0.8])
    plt.xlim([-3, 3])
    plt.xlabel("prior log-odds")
    plt.ylabel("DCF value")
    plt.legend()

def trainWeightedLogRegBinary(DTR, LTR, l, pT):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once
    
    wTar = pT / (ZTR>0).sum() # Compute the weights for the two classes
    wNon = (1-pT) / (ZTR<0).sum()

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(functions.mcol(w).T, DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)
        loss[ZTR>0] *= wTar # Apply the weights to the loss computations
        loss[ZTR<0] *= wNon

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        G[ZTR > 0] *= wTar # Apply the weights to the gradient computations
        G[ZTR < 0] *= wNon
        
        GW = (functions.mrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0]+1))[0]
    #print ("Weighted Log-reg (pT %e) - lambda = %e - J*(w, b) = %e" % (pT, l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]

def apply_kfold(scores, labels, K, pT):
    """
    This function aims at applying k-fold to an array of scores in order to train K different calibration models.
    The scores and the labels should be provided as parameters. The labels parameter should contain the correct labels
    corrensponding to the scores. K parameter is the number of folds required. The function returns a tuple 
    (calibrated scores, labels to corresponding to the scores), where the calibarated scores and the labels
    are both a stack of calibrated scores and labels obtained for a single model
    """


    folds = []
    SCAL = numpy.array([], dtype=numpy.float64)
    LCAL = numpy.array([], dtype=int)
    for idx in range(K):
        folds.append((scores[idx::K], labels[idx::K]))
    for i in range(len(folds)-1, -1, -1): #the index refers to the fold to be left out
        scores_folds = [] #scores training samples for the Mi model
        labels_folds = [] #correct training samples labels for the Mi model
        scores_left_out_fold = folds[i][0] #fold to be left out during this iteration
        labels_left_out_fold = folds[i][1] #fold to be left out during this iteration
        for j in range(0, 5):
            if(j != i): #we take all the folds but the one to be left out
                scores_folds.append(folds[j][0])
                labels_folds.append(folds[j][1])
        scores_folds = numpy.array(scores_folds).flatten() #get a single array from a list of array        
        labels_folds = numpy.array(labels_folds).flatten() #get a single array from a list of array       
        alpha, beta = progetto6.trainWeightedLoReg(functions.mrow(scores_folds), labels_folds, 0.0, pT) #train the Mi model
        calibrated_scores_left_out_fold = alpha @ functions.mrow (scores_left_out_fold) + (beta - numpy.log(pT/(1-pT))) #calibrate the scores
        SCAL = numpy.hstack([SCAL, calibrated_scores_left_out_fold]) #add the calibrated scores to the list to be return
        LCAL = numpy.hstack([LCAL, labels_left_out_fold]) #add the labels corresponding to calibrated scores to the list to be return
    return SCAL, LCAL

def apply_kfold_n_dimensions(scores, labels, K, pT):
    """
    This function aims at applying k-fold to an array of scores in order to train K different calibration models.
    The scores and the labels should be provided as parameters. The labels parameter should contain the correct labels
    corrensponding to the scores. K parameter is the number of folds required. The function returns a tuple 
    (calibrated scores, labels to corresponding to the scores), where the calibarated scores and the labels
    are both a stack of calibrated scores and labels obtained for a single model
    """


    folds = []
    SCAL = numpy.array([], dtype=numpy.float64)
    LCAL = numpy.array([], dtype=int)
    for idx in range(K):
        folds.append((scores[:, idx::K], labels[idx::K]))
    for i in range(len(folds)-1, -1, -1): #the index refers to the fold to be left out
        scores_folds = [] #scores training samples for the Mi model
        labels_folds = [] #correct training samples labels for the Mi model
        scores_left_out_fold = folds[i][0] #fold to be left out during this iteration
        labels_left_out_fold = folds[i][1] #fold to be left out during this iteration
        for j in range(0, 5):
            if(j != i): #we take all the folds but the one to be left out
                scores_folds.append(folds[j][0])
                labels_folds.append(folds[j][1])
        scores_folds = numpy.hstack(scores_folds) #get a single array from a list of array        
        labels_folds = numpy.array(labels_folds).flatten() #get a single array from a list of array       
        alpha, beta = progetto6.trainWeightedLoReg(scores_folds, labels_folds, 0.0, pT) #train the Mi model
        calibrated_scores_left_out_fold = alpha @ scores_left_out_fold + (beta - numpy.log(pT/(1-pT))) #calibrate the scores
        SCAL = numpy.hstack([SCAL, calibrated_scores_left_out_fold]) #add the calibrated scores to the list to be return
        LCAL = numpy.hstack([LCAL, labels_left_out_fold]) #add the labels corresponding to calibrated scores to the list to be return
    return SCAL, LCAL





if __name__ == '__main__':
    D, L = loadData.load('trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = functions.split_training_test_dataset(D, L)

    pT = 0.1

    '''# --- PRIMA COMPARAZIONE SENZA K-FOLD ---
    # --- Quadratic logistic regression ---
    print('-'*40)
    print('Quadratic logistic regression')
    _lambda = 0.03162277660168379
    DVAL_expanded = progetto6.quadratic_feature_expansion(DVAL)
    DTR_expanded = progetto6.quadratic_feature_expansion(DTR)
    w, b = progetto6.trainLogReg(DTR_expanded, LTR, _lambda)   
    Sval = w.T @ DVAL_expanded + b  
    emp_prior = (LTR == 1).sum() / float(LTR.size)
    scores_quad_log_reg = Sval - numpy.log(emp_prior / (1-emp_prior))
    bayes_error_plots(scores_quad_log_reg, LVAL, "Quad-Log-Reg", ["b", "b"]) 

    # --- Non-linear SVM, RBF kernel --- 
    print('-'*40)
    print('Non-Linear SVM, RBF kernel')
    gamma = numpy.exp(-2)
    C = 3.162278e+01
    eps = 1 # RBF kernel does not account for the bias term
    fScore = progetto7.train_dual_SVM_kernel(DTR, LTR, C, progetto7.rbfKernel(gamma), eps)
    scores_svm_rbf = fScore(DVAL)
    bayes_error_plots(scores_svm_rbf, LVAL, "SVM, RBF kernel", ["m", "m"])

    # --- GMM with diagonal covariance matrix ---
    print('-'*40)
    print('GMM with diagonal covariance matrix')
    D0 = DTR[:, LTR == 0]
    D1 = DTR[:, LTR == 1]          
    numComponents0 = 8
    numComponents1 = 32
    covType='Diagonal'
    gmm0 = gmm.train_GMM_LBG_EM(D0, numComponents0, covType, psiEig=0.01, verbose=False)
    gmm1 = gmm.train_GMM_LBG_EM(D1, numComponents1, covType, psiEig=0.01, verbose=False)
    scores_gmm = progetto8.logpdf_GMM(DVAL, gmm1) - progetto8.logpdf_GMM(DVAL, gmm0)
    bayes_error_plots(scores_gmm, LVAL, "GMM, Diag-Cov-Matr", ["orange", "orange"]) 

    plt.show()'''



    # --- COMPARAZIONE CON K-FOLD --- 
    # --- Quadratic logistic regression ---
    print('-'*40)
    print('Quadratic logistic regression')
    _lambda = 0.03162277660168379
    DVAL_expanded = progetto6.quadratic_feature_expansion(DVAL)
    DTR_expanded = progetto6.quadratic_feature_expansion(DTR)
    w, b = progetto6.trainLogReg(DTR_expanded, LTR, _lambda)   # calcolo i parametri del modello, w e b
    Sval = w.T @ DVAL_expanded + b  
    emp_prior = (LTR == 1).sum() / float(LTR.size)
    scores_quad_log_reg = Sval - numpy.log(emp_prior / (1-emp_prior))
    

    #Raw scores
    min_dcf1 = min_DCF(pT, 1.0, 1.0, scores_quad_log_reg, LVAL)
    act_dcf1 = DCF_normalized(pT, 1.0, 1.0, scores_quad_log_reg, LVAL)
    print("Quadratic Logistic Regression, Raw scores\n\tMinimum DCF: %.3f\tActual DCF: %.3f" %(min_dcf1, act_dcf1))

    #K-fold for calibration
    KFOLD = 5
    SCAL1 = []
    LCAL1 = []
    best_SCAL1 = []
    best_LCAL1 = []
    best_prior = 0
    best_act = 100
    for prior in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        SCAL1, LCAL1 = apply_kfold(scores_quad_log_reg, LVAL, KFOLD, prior) #use different priors, we found that the best one is 0.5
        min_dcf1 = min_DCF(pT, 1.0, 1.0, SCAL1, LCAL1)
        act_dcf1 = DCF_normalized(pT, 1.0, 1.0, SCAL1, LCAL1)
        if(act_dcf1 < best_act):
            best_SCAL1 = SCAL1
            best_LCAL1 = LCAL1
            best_act = act_dcf1
            best_prior = prior
        print("Quadratic Logistic Regression, Calibrated scores, Calibration model prior: %.1f" %(prior))
        print("\tMinimum DCF: %.4f\tActual DCF: %.4f" %(min_dcf1, act_dcf1))
    print("Best prior: ", best_prior)
    bayes_error_plots(best_SCAL1, best_LCAL1, "Quad-Log-Reg", ["b", "b"]) #unlock all 3 bayes error plots


    # --- Non-linear SVM, RBF kernel --- 
    print('-'*40)
    print('Non-Linear SVM, RBF kernel')
    gamma = numpy.exp(-2)
    C = 3.162278e+01
    eps = 1 # RBF kernel does not account for the bias term
    fScore = progetto7.train_dual_SVM_kernel(DTR, LTR, C, progetto7.rbfKernel(gamma), eps) #returns the fScore function built
    scores_svm_rbf = fScore(DVAL)

    #Raw scores
    min_dcf2 = min_DCF(pT, 1.0, 1.0, scores_svm_rbf, LVAL)
    act_dcf2 = DCF_normalized(pT, 1.0, 1.0, scores_svm_rbf, LVAL)
    print("Support Vector Machine, RBF Kernel, Raw scores\n\tMinimum DCF: %.3f\tActual DCF: %.3f" %(min_dcf2, act_dcf2))

    #K-fold for calibration
    KFOLD = 5
    SCAL2 = []
    LCAL2 = []
    best_SCAL2 = []
    best_LCAL2 = []
    best_prior = 0
    best_act = 100
    for prior in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        SCAL2, LCAL2 = apply_kfold(scores_svm_rbf, LVAL, KFOLD, prior) #use different priors, we found that the best one is 0.5
        min_dcf2 = progetto6.compute_minDCF(SCAL2, LCAL2, pT, 1.0, 1.0)
        act_dcf2 = DCF_normalized(pT, 1.0, 1.0, SCAL2, LCAL2)
        if(act_dcf2 < best_act):
            best_SCAL2 = SCAL2
            best_LCAL2 = LCAL2
            best_act = act_dcf2
            best_prior = prior
        print("Support Vector Machine, RBF Kernel, Calibrated scores, Calibration model prior: %.1f" %(prior))
        print("\tMinimum DCF: %.4f\tActual DCF: %.4f" %(min_dcf2, act_dcf2))
    print("Best prior: ", best_prior)
    #bayes_error_plots(best_SCAL2, best_LCAL2, "SVM, RBF kernel", ["m", "m"])


    # --- GMM with diagonal covariance matrix ---
    print('-'*40)
    print('GMM with diagonal covariance matrix')
    D0 = DTR[:, LTR == 0]
    D1 = DTR[:, LTR == 1]          
    numComponents0 = 8
    numComponents1 = 32
    covType='Diagonal'
    gmm0 = gmm.train_GMM_LBG_EM(D0, numComponents0, covType, psiEig=0.01, verbose=False)
    gmm1 = gmm.train_GMM_LBG_EM(D1, numComponents1, covType, psiEig=0.01, verbose=False)
    scores_gmm = progetto8.logpdf_GMM(DVAL, gmm1) - progetto8.logpdf_GMM(DVAL, gmm0)
    

    #Raw scores
    min_dcf3 = min_DCF(pT, 1.0, 1.0, scores_gmm, LVAL)
    act_dcf3 = DCF_normalized(pT, 1.0, 1.0, scores_gmm, LVAL)
    print("Gaussian Mixture Model, Diagonal Covariance matrix, Raw scores\n\tMinimum DCF: %.3f\tActual DCF: %.3f" %(min_dcf3, act_dcf3))

    #K-fold for calibration
    KFOLD = 5
    SCAL3 = []
    LCAL3 = []
    best_SCAL3 = []
    best_LCAL3 = []
    best_prior = 0
    best_act = 100
    for prior in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        SCAL3, LCAL3 = apply_kfold(scores_gmm, LVAL, KFOLD, prior) #use different priors, we found that the best one is 0.5
        min_dcf3 = progetto6.compute_minDCF(SCAL3, LCAL2, pT, 1.0, 1.0)
        act_dcf3 = DCF_normalized(pT, 1.0, 1.0, SCAL3, LCAL3)
        if(act_dcf3 < best_act):
            best_SCAL3 = SCAL3
            best_LCAL3 = LCAL3
            best_act = act_dcf3
            best_prior = prior
        print("Gaussian Mixture Model, Diagonal Covariance matrix, Calibrated scores, Calibration model prior: %.1f" %(prior))
        print("\tMinimum DCF: %.4f\tActual DCF: %.4f" %(min_dcf3, act_dcf3))
    print("Best prior: ", best_prior)
    #bayes_error_plots(best_SCAL3, best_LCAL3, "GMM, Diag-Cov-Matr", ["orange", "orange"])
    #plt.show()


    # --- Score-level fusion ---
    print('-'*40)
    print('Score level fusion: GMM + SVM + QLR')
    scores_fusion = numpy.vstack([scores_quad_log_reg, scores_svm_rbf, scores_gmm])
    labels_fusion = LVAL

    KFOLD = 5 #K = 5
    SCAL = []
    LCAL = []
    best_SCAL = []
    best_LCAL = []
    best_prior = 0
    best_act = 100
    for prior in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        SCAL, LCAL = apply_kfold_n_dimensions(scores_fusion, labels_fusion, KFOLD, prior)
        min_dcf = progetto6.compute_minDCF(SCAL, LCAL, pT, 1.0, 1.0)
        act_dcf = DCF_normalized(pT, 1.0, 1.0, SCAL, LCAL)
        if(act_dcf < best_act):
            best_SCAL = SCAL
            best_LCAL = LCAL
            best_act = act_dcf
            best_prior = prior
        print("Score-level fusion, K-fold, Calibration model prior: %.1f\tMinimum DCF: %.4f\tActual DCF: %.4f" %(prior, min_dcf, act_dcf))
    print("Best prior: ", best_prior)
    #bayes_error_plots(best_SCAL, best_LCAL, "Fusion", ["green", "green"])


    print('-'*40)
    print('Score level fusion: GMM + QLR')
    scores_fusion = numpy.vstack([scores_quad_log_reg, scores_gmm])
    labels_fusion = LVAL

    KFOLD = 5 #K = 5
    SCAL = []
    LCAL = []
    best_SCAL = []
    best_LCAL = []
    best_prior = 0
    best_act = 100
    for prior in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        SCAL, LCAL = apply_kfold_n_dimensions(scores_fusion, labels_fusion, KFOLD, prior)
        min_dcf = min_DCF(pT, 1.0, 1.0, SCAL, LCAL)
        act_dcf = DCF_normalized(pT, 1.0, 1.0, SCAL, LCAL)
        if(act_dcf < best_act):
            best_SCAL = SCAL
            best_LCAL = LCAL
            best_act = act_dcf
            best_prior = prior
        print("Score-level fusion, K-fold, Calibration model prior: %.1f\tMinimum DCF: %.4f\tActual DCF: %.4f" %(prior, min_dcf, act_dcf))
    print("Best prior: ", best_prior)
    #bayes_error_plots(best_SCAL, best_LCAL, "Fusion GMM + QLR", ["m", "m"])


    print('-'*40)
    print('Score level fusion: GMM + SVM')
    scores_fusion = numpy.vstack([scores_svm_rbf, scores_gmm])
    labels_fusion = LVAL

    KFOLD = 5 #K = 5
    SCAL = []
    LCAL = []
    best_SCAL = []
    best_LCAL = []
    best_prior = 0
    best_act = 100
    for prior in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        SCAL, LCAL = apply_kfold_n_dimensions(scores_fusion, labels_fusion, KFOLD, prior)
        min_dcf = min_DCF(pT, 1.0, 1.0, SCAL, LCAL)
        act_dcf = DCF_normalized(pT, 1.0, 1.0, SCAL, LCAL)
        if(act_dcf < best_act):
            best_SCAL = SCAL
            best_LCAL = LCAL
            best_act = act_dcf
            best_prior = prior
        print("Score-level fusion, K-fold, Calibration model prior: %.1f\tMinimum DCF: %.4f\tActual DCF: %.4f" %(prior, min_dcf, act_dcf))
    print("Best prior: ", best_prior)
    #bayes_error_plots(best_SCAL, best_LCAL, "Fusion GMM + SVM", ["y", "y"])

    plt.show()
    



    # --- EVALUATION ---
    DEVAL, LEVAL = loadData.load('evalData.txt')


    # QLR
    DEVAL_expanded = progetto6.quadratic_feature_expansion(DEVAL)
    prior_calibration_qlr = 0.6
    alpha, beta = trainWeightedLogRegBinary(functions.mrow(scores_quad_log_reg), LVAL, 0, prior_calibration_qlr)
    Seval = w.T @ DEVAL_expanded + b  
    SEVAL_QLR = Seval - numpy.log(emp_prior / (1-emp_prior))
    calibrated_scores_qlr = alpha @ functions.mrow(SEVAL_QLR) + (beta - numpy.log(prior_calibration_qlr/(1 - prior_calibration_qlr)))
    min_dcf = min_DCF(pT, 1.0, 1.0, calibrated_scores_qlr, LEVAL)
    act_dcf = DCF_normalized(pT, 1.0, 1.0, calibrated_scores_qlr, LEVAL)
    print('Quadratic Logistic Regression\tminDCF: %.4f\tactDCF: %.4f' % (min_dcf, act_dcf))
    #bayes_error_plots(calibrated_scores_qlr, LEVAL, "QLR", ["r", "r"])

    # GMM 
    prior_calibration_gmm = 0.8
    alpha, beta = trainWeightedLogRegBinary(functions.mrow(scores_gmm), LVAL, 0, prior_calibration_gmm)
    SEVAL_GMM = progetto8.logpdf_GMM(DEVAL, gmm1) - progetto8.logpdf_GMM(DEVAL, gmm0)
    calibrated_scores_gmm = alpha @ functions.mrow(SEVAL_GMM) + (beta - numpy.log(prior_calibration_gmm/(1 - prior_calibration_gmm)))
    min_dcf = min_DCF(pT, 1.0, 1.0, calibrated_scores_gmm, LEVAL)
    act_dcf = DCF_normalized(pT, 1.0, 1.0, calibrated_scores_gmm, LEVAL)
    print('GMM\tminDCF: %.4f\tactDCF: %.4f' % (min_dcf, act_dcf))
    #bayes_error_plots(calibrated_scores_gmm, LEVAL, "GMM", ["b", "b"])

    # SVM
    prior_calibration_svm = 0.3
    alpha, beta = trainWeightedLogRegBinary(functions.mrow(scores_svm_rbf), LVAL, 0, prior_calibration_svm)
    SEVAL_SVM = fScore(DEVAL)
    calibrated_scores_svm = alpha @ functions.mrow(SEVAL_SVM) + (beta - numpy.log(prior_calibration_svm/(1 - prior_calibration_svm)))
    min_dcf = min_DCF(pT, 1.0, 1.0, calibrated_scores_svm, LEVAL)
    act_dcf = DCF_normalized(pT, 1.0, 1.0, calibrated_scores_svm, LEVAL)
    print('SVM\tminDCF: %.4f\tactDCF: %.4f' % (min_dcf, act_dcf))
    #bayes_error_plots(calibrated_scores_svm, LEVAL, "SVM", ["m", "m"])

    # QLR + GMM
    prior_calibration_qlr_gmm = 0.4
    gmm0 = gmm.train_GMM_LBG_EM(D0, 6, covType, psiEig=0.01, verbose=False)
    gmm1 = gmm.train_GMM_LBG_EM(D1, 32, covType, psiEig=0.01, verbose=False)
    scores_gmm = progetto8.logpdf_GMM(DVAL, gmm1) - progetto8.logpdf_GMM(DVAL, gmm0)
    score_fusion_qlr_gmm = numpy.vstack([scores_quad_log_reg, scores_gmm])
    alpha, beta = trainWeightedLogRegBinary(score_fusion_qlr_gmm, LVAL, 0, prior_calibration_qlr_gmm)
    SEVAL_fusion_qlr_gmm = numpy.vstack([SEVAL_QLR, SEVAL_GMM])
    calibrated_scores_fusion_qlr_gmm = alpha @ SEVAL_fusion_qlr_gmm + (beta - numpy.log(prior_calibration_qlr_gmm/(1 - prior_calibration_qlr_gmm)))
    min_dcf = min_DCF(pT, 1.0, 1.0, calibrated_scores_fusion_qlr_gmm, LEVAL)
    act_dcf = DCF_normalized(pT, 1.0, 1.0, calibrated_scores_fusion_qlr_gmm, LEVAL)
    print('Fusion QLR + GMM (6, 32)\tminDCF: %.4f\tactDCF: %.4f' % (min_dcf, act_dcf))
    #bayes_error_plots(calibrated_scores_fusion_qlr_gmm, LEVAL, "Fusion QLR + GMM", ["g", "g"])

    #plt.show()

    # GMM (8, 16)
    prior_calibration_gmm = 0.8
    gmm0 = gmm.train_GMM_LBG_EM(D0, 8, covType, psiEig=0.01, verbose=False)
    gmm1 = gmm.train_GMM_LBG_EM(D1, 16, covType, psiEig=0.01, verbose=False)
    scores_gmm = progetto8.logpdf_GMM(DVAL, gmm1) - progetto8.logpdf_GMM(DVAL, gmm0)
    alpha, beta = trainWeightedLogRegBinary(functions.mrow(scores_gmm), LVAL, 0, prior_calibration_gmm)
    SEVAL_GMM = progetto8.logpdf_GMM(DEVAL, gmm1) - progetto8.logpdf_GMM(DEVAL, gmm0)
    calibrated_scores_gmm = alpha @ functions.mrow(SEVAL_GMM) + (beta - numpy.log(prior_calibration_gmm/(1 - prior_calibration_gmm)))
    min_dcf = min_DCF(pT, 1.0, 1.0, calibrated_scores_gmm, LEVAL)
    act_dcf = DCF_normalized(pT, 1.0, 1.0, calibrated_scores_gmm, LEVAL)
    print('GMM with modified parameters\tminDCF: %.4f\tactDCF: %.4f' % (min_dcf, act_dcf))
    #bayes_error_plots(calibrated_scores_gmm, LEVAL, "GMM(8, 16)", ["g", "g"])

    # QLR + GMM(8, 16):
    prior_calibration_qlr_gmm = 0.4
    scores_gmm = progetto8.logpdf_GMM(DVAL, gmm1) - progetto8.logpdf_GMM(DVAL, gmm0)
    score_fusion_qlr_gmm = numpy.vstack([scores_quad_log_reg, scores_gmm])
    alpha, beta = trainWeightedLogRegBinary(score_fusion_qlr_gmm, LVAL, 0, prior_calibration_qlr_gmm)
    SEVAL_fusion_qlr_gmm = numpy.vstack([SEVAL_QLR, SEVAL_GMM])
    calibrated_scores_fusion_qlr_gmm = alpha @ SEVAL_fusion_qlr_gmm + (beta - numpy.log(prior_calibration_qlr_gmm/(1 - prior_calibration_qlr_gmm)))
    min_dcf = min_DCF(pT, 1.0, 1.0, calibrated_scores_fusion_qlr_gmm, LEVAL)
    act_dcf = DCF_normalized(pT, 1.0, 1.0, calibrated_scores_fusion_qlr_gmm, LEVAL)
    print('Fusion QLR + GMM (8, 16)\tminDCF: %.4f\tactDCF: %.4f' % (min_dcf, act_dcf))

    # QLR + GMM(8, 32):
    prior_calibration_qlr_gmm = 0.4
    gmm0 = gmm.train_GMM_LBG_EM(D0, 8, covType, psiEig=0.01, verbose=False)
    gmm1 = gmm.train_GMM_LBG_EM(D1, 32, covType, psiEig=0.01, verbose=False)
    scores_gmm = progetto8.logpdf_GMM(DVAL, gmm1) - progetto8.logpdf_GMM(DVAL, gmm0)
    score_fusion_qlr_gmm = numpy.vstack([scores_quad_log_reg, scores_gmm])
    alpha, beta = trainWeightedLogRegBinary(score_fusion_qlr_gmm, LVAL, 0, prior_calibration_qlr_gmm)
    SEVAL_fusion_qlr_gmm = numpy.vstack([SEVAL_QLR, SEVAL_GMM])
    calibrated_scores_fusion_qlr_gmm = alpha @ SEVAL_fusion_qlr_gmm + (beta - numpy.log(prior_calibration_qlr_gmm/(1 - prior_calibration_qlr_gmm)))
    min_dcf = min_DCF(pT, 1.0, 1.0, calibrated_scores_fusion_qlr_gmm, LEVAL)
    act_dcf = DCF_normalized(pT, 1.0, 1.0, calibrated_scores_fusion_qlr_gmm, LEVAL)
    print('Fusion QLR + GMM (8, 32)\tminDCF: %.4f\tactDCF: %.4f' % (min_dcf, act_dcf))

    