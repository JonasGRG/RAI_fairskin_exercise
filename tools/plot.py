
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
import numpy as np
import seaborn as sns

sns.set_theme(style='whitegrid')



def heatmap_joint_dist(y_train,a_train,y_test,a_test):
    fig,axes = plt.subplots(1,2,figsize=(10,4),dpi=100)

    # train
    y_labels = y_train.cpu().detach().numpy() 
    a_labels = a_train.cpu().detach().numpy()  
    df = pd.DataFrame({'Y': y_labels, 'A': a_labels})
    ct = pd.crosstab(df['Y'], df['A'])

    # Plotting the heatmap
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", cbar=True,ax=axes[0])
    axes[0].set_title('train set')

    # test
    y_labels = y_test.cpu().detach().numpy() 
    a_labels = a_test.cpu().detach().numpy()  
    df = pd.DataFrame({'Y': y_labels, 'A': a_labels})
    ct = pd.crosstab(df['Y'], df['A'])

    # Plotting the heatmap
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", cbar=True,ax=axes[1])
    axes[1].set_title('test set')


    plt.show()
    return 


def plotROC():
    '''
    Draw ROC curves based on sensitive attributes
    '''

    return

