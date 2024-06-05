import numpy
import scipy
import functions
import loadData
import plots
import matplotlib
import matplotlib.pyplot
import sklearn.datasets



def compute_predictions(D, class_prior_prob, llr, threshold):
    PVAL = numpy.zeros(DVAL.shape[1], dtype = numpy.int32)
    class_prior_prob = [0.5, 0.5]   # class prior probabilities

    threshold = -numpy.log(class_prior_prob[1]/class_prior_prob[0])     # compute the threshold as -log( P(C == 1)/P(C == 0) )

    PVAL[llr >= threshold] = 1
    PVAL[llr < threshold] = 0

    return PVAL


def compute_error_rate(P, L):
    return ((P != L).sum()/float(L.size))*100


def compute_optimal_bayes_binary_llr(llr, prior, Cfn, Cfp):
    threshold = -numpy.log((prior * Cfn) / ((1-prior)*Cfp))   # uso la formula per calcolare la threshold e seleziono poi i llr > threshold

    return numpy.int32(llr > threshold)


def compute_confusion_matrix(P, L):

    nclasses = L.max() + 1
    conf_mat = numpy.zeros((nclasses, nclasses), dtype = numpy.int32)

    for i in range(P.shape[0]):    # uso i valori contenuti in P ed L come indici per creare la confusion matrix

        index_pred = P[i]   # prendo il valore contenuto in P
        index_class = L[i]  # prendo il valore contenuto in L

        conf_mat[index_pred][index_class] += 1

    return conf_mat


def compute_empirical_bayes_risk_binary(confusion_matrix, prior, Cfn, Cfp, normalize=True):
    Pfn = confusion_matrix[0][1] / (confusion_matrix[0][1] + confusion_matrix[1][1])   # false negative rate
    Pfp = confusion_matrix[1][0] / (confusion_matrix[1][0] + confusion_matrix[0][0])   # false positive rate
    DCFu = (prior*Cfn*Pfn) + (1 - prior)*Cfp*Pfp

    if normalize:
        return DCFu / numpy.minimum(prior * Cfn, (1-prior)*Cfp)

    return DCFu


def compute_minDCF(llr, labels, prior, Cfn, Cfp, returnThreshold=False):

    llr_sorted = llr

    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), llr_sorted, numpy.array([numpy.inf])])
    DCF_min = None
    DCF_th = None

    for th in thresholds:

        predicted_labels = numpy.int32(llr > th)
        conf_matrix = compute_confusion_matrix(predicted_labels, labels)
        DCF_normalized = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp, normalize=True)

        if DCF_min is None or DCF_normalized < DCF_min:
            DCF_min = DCF_normalized
            DCF_th = th

    if returnThreshold:
        return DCF_min, DCF_th
    
    return DCF_min


def compute_effective_prior(threshold):
    return 1.0 / (1.0 + numpy.exp(-threshold))

#compute optimal Bayes decisions for the matrix of class posteriors
def compute_optimal_Bayes(posterior, cost_matrix):
    expected_costs = cost_matrix @ posterior
    return numpy.argmin(expected_costs, axis = 0)



def bayes_error_plots(llrs, LVAL, tags):

    colors = ['b','g','r','c','m','y']

    effPriorLogOdds = numpy.linspace(-4, 4, 21)     # creo una serie di punti equispaziati (21, dato che e' il numero di punti che valutiamo con la DCF)
    effPriors = 1.0 / (1.0 + numpy.exp(-effPriorLogOdds))


    for i in range(len(llrs)):

        actual_DCF = []
        min_DCF = []
        Cfn = 1.0
        Cfp = 1.0

        c = i*2

        llr = llrs[i]

        for effPrior in effPriors:
            predictions = compute_optimal_bayes_binary_llr(llr, effPrior, Cfn, Cfp)
            conf_matrix = compute_confusion_matrix(predictions, LVAL)

            actualDCF = compute_empirical_bayes_risk_binary(conf_matrix, effPrior, Cfn, Cfp)    # NORMALIZED DCF = ACTUAL DCF
            minDCF = compute_minDCF(llr, LVAL, effPrior, Cfn, Cfp)

            actual_DCF.append(actualDCF)
            min_DCF.append(minDCF)


        matplotlib.pyplot.plot(effPriorLogOdds, actual_DCF, label=tags[i]+' DCF', color = colors[c])
        matplotlib.pyplot.plot(effPriorLogOdds, min_DCF, label=tags[i]+' minDCF', color = colors[c+1])
        matplotlib.pyplot.ylim([0, 1.1])
        matplotlib.pyplot.xlim([-4, 4])
        matplotlib.pyplot.legend()
        matplotlib.pyplot.xlabel('prior log-odds')
        matplotlib.pyplot.ylabel('DCF value')

    matplotlib.pyplot.show()












########################################################
#                                                      #
#-------------------------MAIN-------------------------#
#                                                      #
########################################################

if __name__ == '__main__':
    D, L = loadData.load('trainData.txt')      # get the data and labels from the dataset

    class_prior_prob = [0.5, 0.5]   # class prior probabilities
    threshold = -numpy.log(class_prior_prob[1]/class_prior_prob[0])     # compute the threshold as -log( P(C == 1)/P(C == 0) )

    (DTR, LTR), (DVAL, LVAL) = functions.split_training_test_dataset(D, L)  # obtain training and validation data


    # ----- MVG -----
    parameters = functions.compute_parameters_MVG(DTR, LTR)   # compute training parameters with MVG model
    llr = functions.compute_llr(DVAL, parameters)

    # predictions:
    PVAL = compute_predictions(DVAL, class_prior_prob, llr, threshold)
    #print('MVG model -  classification error rate (threshold: ',threshold, '): ', compute_error_rate(PVAL, LVAL), '%')

    print('-'*40)
    print('MVG classifier - prior probabilities', class_prior_prob)
    for prior, Cfn, Cfp in [(0.5, 1.0, 1.0), (0.9, 1.0, 1.0), (0.1, 1.0, 1.0), (0.5, 1.0, 9.0), (0.5, 9.0, 1.0)]:  
        print()
        print('Prior:', prior, '- Cfn:', Cfn, '- Cfp:', Cfp)

        predictions_binary = compute_optimal_bayes_binary_llr(llr, prior, Cfn, Cfp)   
        conf_matrix = compute_confusion_matrix(predictions_binary, LVAL)
        #print(conf_matrix)

        #bayes_risk = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp, normalize=False)   
        #print('Bayes risk:', round(bayes_risk, 5))

        normalized_bayes = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp) 
        print('Actual DCF:', round(normalized_bayes, 5)) 

        DCF_min, threshold_min = compute_minDCF(llr, LVAL, prior, Cfn, Cfp, True)
        print('DCF min:', round(DCF_min, 5), '- Threshold:', round(threshold_min, 5))
        print('effective prior:', round(compute_effective_prior(threshold_min), 5))

    # ----- TIED GAUSSIAN -----
    parameters = functions.compute_parameters_tied(DTR, LTR)  # compute training parameters with tied Gaussian model
    llr = functions.compute_llr(DVAL, parameters)
    # predictions:
    PVAL = compute_predictions(DVAL, class_prior_prob, llr, threshold)

    print('-'*40)
    print('Tied Gaussian classifier - prior probabilities', class_prior_prob)
    for prior, Cfn, Cfp in [(0.5, 1.0, 1.0), (0.9, 1.0, 1.0), (0.1, 1.0, 1.0), (0.5, 1.0, 9.0), (0.5, 9.0, 1.0)]:  
        print()
        print('Prior:', prior, '- Cfn:', Cfn, '- Cfp:', Cfp)

        predictions_binary = compute_optimal_bayes_binary_llr(llr, prior, Cfn, Cfp)   
        conf_matrix = compute_confusion_matrix(predictions_binary, LVAL)
        #print(conf_matrix)

        #bayes_risk = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp, normalize=False)   
        #print('Bayes risk:', round(bayes_risk, 5))

        normalized_bayes = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp) 
        print('Actual DCF:', round(normalized_bayes, 5)) 

        DCF_min, threshold_min = compute_minDCF(llr, LVAL, prior, Cfn, Cfp, True)
        print('DCF min:', round(DCF_min, 5), '- Threshold:', round(threshold_min, 5))
        print('effective prior:', round(compute_effective_prior(threshold_min), 5))


    # ----- NAIVE BAYES GAUSSIAN -----
    parameters = functions.compute_parameters_naive_bayes(DTR, LTR)
    llr = functions.compute_llr(DVAL, parameters)
    # predictions
    PVAL = compute_predictions(DVAL, class_prior_prob, llr, threshold)

    print('-'*40)
    print('Naive Bayes classifier - prior probabilities', class_prior_prob)
    for prior, Cfn, Cfp in [(0.5, 1.0, 1.0), (0.9, 1.0, 1.0), (0.1, 1.0, 1.0), (0.5, 1.0, 9.0), (0.5, 9.0, 1.0)]:  
        print()
        print('Prior:', prior, '- Cfn:', Cfn, '- Cfp:', Cfp)

        predictions_binary = compute_optimal_bayes_binary_llr(llr, prior, Cfn, Cfp)   
        conf_matrix = compute_confusion_matrix(predictions_binary, LVAL)
        #print(conf_matrix)

        #bayes_risk = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp, normalize=False)   
        #print('Bayes risk:', round(bayes_risk, 5))

        normalized_bayes = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp) 
        print('Actual DCF:', round(normalized_bayes, 5)) 

        DCF_min, threshold_min = compute_minDCF(llr, LVAL, prior, Cfn, Cfp, True)
        print('DCF min:', round(DCF_min, 5), '- Threshold:', round(threshold_min, 5))
        print('effective prior:', round(compute_effective_prior(threshold_min), 5))


    ########################################################
    #                                                      #
    #---3 APPLICATIONS WITH EFFECTIVE PRIOR WITHOUT PCA----#
    #                                                      #
    ########################################################

    print('-'*40)
    print('-'*40)
    print('---------------WITHOUT PCA---------------')
    print('-'*40)
    print('-'*40)
    parameters = functions.compute_parameters_MVG(DTR, LTR)   # compute training parameters with MVG model
    llr = functions.compute_llr(DVAL, parameters)

    # predictions:
    PVAL = compute_predictions(DVAL, class_prior_prob, llr, threshold)
    #print('MVG model -  classification error rate (threshold: ',threshold, '): ', compute_error_rate(PVAL, LVAL), '%')

    
    print('MVG classifier - prior probabilities', class_prior_prob)
    for prior, Cfn, Cfp in [(0.1, 1.0, 1.0), (0.5, 1.0, 1.0), (0.9, 1.0, 1.0)]:  
        print()
        print('effPrior:', prior, '- Cfn:', Cfn, '- Cfp:', Cfp)

        predictions_binary = compute_optimal_bayes_binary_llr(llr, prior, Cfn, Cfp)   
        conf_matrix = compute_confusion_matrix(predictions_binary, LVAL)
        #print(conf_matrix)

        #bayes_risk = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp, normalize=False)   
        #print('Bayes risk:', round(bayes_risk, 5))

        normalized_bayes = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp) 
        print('Actual DCF:', round(normalized_bayes, 5)) 

        DCF_min, threshold_min = compute_minDCF(llr, LVAL, prior, Cfn, Cfp, True)
        print('DCF min:', round(DCF_min, 5), '- Threshold:', round(threshold_min, 5))
        print('DCF difference =', round(normalized_bayes-DCF_min, 5), '-', round(((normalized_bayes-DCF_min)/DCF_min)*100, 5), '%')

    print('-'*40)
    # ----- TIED GAUSSIAN -----
    parameters = functions.compute_parameters_tied(DTR, LTR)  # compute training parameters with tied Gaussian model
    llr = functions.compute_llr(DVAL, parameters)
    # predictions:
    PVAL = compute_predictions(DVAL, class_prior_prob, llr, threshold)

    
    print('Tied Gaussian classifier - prior probabilities', class_prior_prob)
    for prior, Cfn, Cfp in [(0.1, 1.0, 1.0), (0.5, 1.0, 1.0), (0.9, 1.0, 1.0)]:  
        print()
        print('effPrior:', prior, '- Cfn:', Cfn, '- Cfp:', Cfp)

        predictions_binary = compute_optimal_bayes_binary_llr(llr, prior, Cfn, Cfp)   
        conf_matrix = compute_confusion_matrix(predictions_binary, LVAL)
        #print(conf_matrix)

        #bayes_risk = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp, normalize=False)   
        #print('Bayes risk:', round(bayes_risk, 5))

        normalized_bayes = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp) 
        print('Actual DCF:', round(normalized_bayes, 5)) 

        DCF_min, threshold_min = compute_minDCF(llr, LVAL, prior, Cfn, Cfp, True)
        print('DCF min:', round(DCF_min, 5), '- Threshold:', round(threshold_min, 5))
        print('DCF difference =', round(normalized_bayes-DCF_min, 5), '-', round(((normalized_bayes-DCF_min)/DCF_min)*100, 5), '%')

    print('-'*40)
    # ----- NAIVE BAYES GAUSSIAN -----
    parameters = functions.compute_parameters_naive_bayes(DTR, LTR)
    llr = functions.compute_llr(DVAL, parameters)
    # predictions
    PVAL = compute_predictions(DVAL, class_prior_prob, llr, threshold)

    
    print('Naive Bayes classifier - prior probabilities', class_prior_prob)
    for prior, Cfn, Cfp in [(0.1, 1.0, 1.0), (0.5, 1.0, 1.0), (0.9, 1.0, 1.0)]:  
        print()
        print('effPrior:', prior, '- Cfn:', Cfn, '- Cfp:', Cfp)

        predictions_binary = compute_optimal_bayes_binary_llr(llr, prior, Cfn, Cfp)   
        conf_matrix = compute_confusion_matrix(predictions_binary, LVAL)
        #print(conf_matrix)

        #bayes_risk = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp, normalize=False)   
        #print('Bayes risk:', round(bayes_risk, 5))

        normalized_bayes = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp) 
        print('Actual DCF:', round(normalized_bayes, 5)) 

        DCF_min, threshold_min = compute_minDCF(llr, LVAL, prior, Cfn, Cfp, True)
        print('DCF min:', round(DCF_min, 5), '- Threshold:', round(threshold_min, 5))
        print('DCF difference =', round(normalized_bayes-DCF_min, 5), '-', round(((normalized_bayes-DCF_min)/DCF_min)*100, 5), '%')
    print('-'*40)

    ########################################################
    #                                                      #
    #-----3 APPLICATIONS WITH EFFECTIVE PRIOR WITH PCA-----#
    #                                                      #
    ########################################################

    print('-'*40)
    print('-'*40)
    print('-----------------WITH PCA---------------')
    print('-'*40)
    print('-'*40)

    for m in range(1, 7):
        print('m =', m)
        DTR_pca = functions.PCA_matrix(DTR, m) 

        DTRP_pca = DTR_pca.T @ DTR    # project the data over the new subspace
        DVALP_pca = DTR_pca.T @ DVAL
        parameters = functions.compute_parameters_MVG(DTRP_pca, LTR)   # compute training parameters with MVG model
        llr = functions.compute_llr(DVALP_pca, parameters)

        # predictions:
        PVAL = compute_predictions(DVALP_pca, class_prior_prob, llr, threshold)
        #print('MVG model -  classification error rate (threshold: ',threshold, '): ', compute_error_rate(PVAL, LVAL), '%')

        print('-'*40)
        print('MVG classifier - prior probabilities', class_prior_prob, '- PCA m =', m)
        for prior, Cfn, Cfp in [(0.1, 1.0, 1.0), (0.5, 1.0, 1.0), (0.9, 1.0, 1.0)]:  
            print()
            print('Prior:', prior, '- Cfn:', Cfn, '- Cfp:', Cfp)

            predictions_binary = compute_optimal_bayes_binary_llr(llr, prior, Cfn, Cfp)   
            conf_matrix = compute_confusion_matrix(predictions_binary, LVAL)
            #print(conf_matrix)

            #bayes_risk = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp, normalize=False)   
            #print('Bayes risk:', round(bayes_risk, 5))

            normalized_bayes = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp) 
            print('Actual DCF:', round(normalized_bayes, 5)) 

            DCF_min, threshold_min = compute_minDCF(llr, LVAL, prior, Cfn, Cfp, True)
            print('DCF min:', round(DCF_min, 5), '- Threshold:', round(threshold_min, 5))
            print('DCF difference =', round(normalized_bayes-DCF_min, 5), '-', round(((normalized_bayes-DCF_min)/DCF_min)*100, 5), '%')

        # ----- TIED GAUSSIAN -----
        parameters = functions.compute_parameters_tied(DTRP_pca, LTR)  # compute training parameters with tied Gaussian model
        llr = functions.compute_llr(DVALP_pca, parameters)
        # predictions:
        PVAL = compute_predictions(DVALP_pca, class_prior_prob, llr, threshold)

        print('-'*40)
        print('Tied Gaussian classifier - prior probabilities', class_prior_prob, '- PCA m =', m)
        for prior, Cfn, Cfp in [(0.1, 1.0, 1.0), (0.5, 1.0, 1.0), (0.9, 1.0, 1.0)]:  
            print()
            print('Prior:', prior, '- Cfn:', Cfn, '- Cfp:', Cfp)

            predictions_binary = compute_optimal_bayes_binary_llr(llr, prior, Cfn, Cfp)   
            conf_matrix = compute_confusion_matrix(predictions_binary, LVAL)
            #print(conf_matrix)

            #bayes_risk = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp, normalize=False)   
            #print('Bayes risk:', round(bayes_risk, 5))

            normalized_bayes = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp) 
            print('Actual DCF:', round(normalized_bayes, 5)) 

            DCF_min, threshold_min = compute_minDCF(llr, LVAL, prior, Cfn, Cfp, True)
            print('DCF min:', round(DCF_min, 5), '- Threshold:', round(threshold_min, 5))
            print('DCF difference =', round(normalized_bayes-DCF_min, 5), '-', round(((normalized_bayes-DCF_min)/DCF_min)*100, 5), '%')


        # ----- NAIVE BAYES GAUSSIAN -----
        parameters = functions.compute_parameters_naive_bayes(DTRP_pca, LTR)
        llr = functions.compute_llr(DVALP_pca, parameters)
        # predictions
        PVAL = compute_predictions(DVALP_pca, class_prior_prob, llr, threshold)

        print('-'*40)
        print('Naive Bayes classifier - prior probabilities', class_prior_prob, '- PCA m =', m)
        for prior, Cfn, Cfp in [(0.1, 1.0, 1.0), (0.5, 1.0, 1.0), (0.9, 1.0, 1.0)]:  
            print()
            print('Prior:', prior, '- Cfn:', Cfn, '- Cfp:', Cfp)

            predictions_binary = compute_optimal_bayes_binary_llr(llr, prior, Cfn, Cfp)   
            conf_matrix = compute_confusion_matrix(predictions_binary, LVAL)
            #print(conf_matrix)

            #bayes_risk = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp, normalize=False)   
            #print('Bayes risk:', round(bayes_risk, 5))

            normalized_bayes = compute_empirical_bayes_risk_binary(conf_matrix, prior, Cfn, Cfp) 
            print('Actual DCF:', round(normalized_bayes, 5)) 

            DCF_min, threshold_min = compute_minDCF(llr, LVAL, prior, Cfn, Cfp, True)
            print('DCF min:', round(DCF_min, 5), '- Threshold:', round(threshold_min, 5))
            print('DCF difference =', round(normalized_bayes-DCF_min, 5), '-', round(((normalized_bayes-DCF_min)/DCF_min)*100, 5), '%')
        print('-'*40)



    # BEST PCA: M = 1
    m_values = [6, 4, 6]

    DTRP_pcas = []
    DVALP_pcas = []

    for m in m_values:
        DTR_pca = functions.PCA_matrix(DTR, m) 
        DTRP = DTR_pca.T @ DTR    # project the data over the new subspace
        DVALP = DTR_pca.T @ DVAL

        DTRP_pcas.append(DTRP)
        DVALP_pcas.append(DVALP)

    tags = ['MVG', 'Tied', 'Naive']

    llrs = []

    for i in range(3):

        model = tags[i]

        DTRP_pca = DTRP_pcas[i]
        DVALP_pca = DVALP_pcas[i]


        if model == 'MVG':
            parameters = functions.compute_parameters_MVG(DTRP_pca, LTR)   # compute training parameters with MVG model
        
        elif model == 'Tied':
            parameters = functions.compute_parameters_tied(DTRP_pca, LTR)  # compute training parameters with tied Gaussian model

        elif model == 'Naive':
            parameters = functions.compute_parameters_naive_bayes(DTRP_pca, LTR) # compute training parameters with Naive Bayes model
        

        llrs.append(functions.compute_llr(DVALP_pca, parameters))

    
    bayes_error_plots(llrs, LVAL, tags)