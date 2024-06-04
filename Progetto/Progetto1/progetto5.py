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