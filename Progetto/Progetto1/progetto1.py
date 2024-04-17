import numpy
import matplotlib
import matplotlib.pyplot as plt
import plots
import loadData



if __name__ == '__main__':

    #change default font size - comment to use default values
    plt.rc('font', size = 16)
    plt.rc('xtick', labelsize = 16)
    plt.rc('ytick', labelsize = 16)


    D, L = loadData.load('trainData.txt')
    
     
    #feature 1 - feature 2
    plots.plot_histograms("plots_p1", D, L, [0, 1])
    plots.plot_scatter("plots_p1", D, L, [0, 1])

    #feature 3 - feature 4
    plots.plot_histograms("plots_p1", D, L, [2, 3])
    plots.plot_scatter("plots_p1", D, L, [2, 3])

    #feature 5 - feature 6
    plots.plot_histograms("plots_p1", D, L, [4, 5])
    plots.plot_scatter("plots_p1", D, L, [4, 5])
    
    '''
    mu = D.mean(1).reshape((D.shape[0], 1))
    var = D.var(1)
    print('Mean =\n {} \n\n Variance = {}\n'.format(mu, var))
    '''

    for cls in [0, 1]:
        print('class', cls)
        Dcls = D[:, L == cls]
        mu = Dcls.mean(1).reshape(Dcls.shape[0], 1)
        print('Mean:')
        print(mu)
        
        var = Dcls.var(1)
        print('Variance', var)
        print()
    