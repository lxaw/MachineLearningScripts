from tkinter import W
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X,y,classifier, resolution=0.02):
    # set up colors and markers
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # set up decision boundary graph
    x1_min, x1_max = X[:,0].min()-1, X[:,0].max() +1
    x2_min, x2_max = X[:,1].min()-1, X[:,1].max() +1

    # generation of grid points
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
        np.arange(x2_min,x2_max,resolution)
    )

    # for each feature, prepare an array (1d)
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    # change predictions to data size array 
    Z = Z.reshape(xx1.shape)
    # grid point plot
    plt.contourf(xx1,xx2,Z,alpha=0.3,cmap=cmap)
    # set the bounds
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    # plot classes and training data
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],
            y=X[y==cl,1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=cl,
            edgecolor='black'
        )




class Perceptron(object):
    """
    eta: learning rate
    n_iter: training data's amount of training runs
    random_state: randomly initialized weights
    w_: 1d list of weights
    errors_: count of each epoch's updates 
    """

    def __init__(self, eta=0.01,n_iter=50,random_state = 1):
        self.eta = eta
        self.n_iter =  n_iter
        self.random_state = random_state

    def fit(self,X,y):
        """
        Fit the training data
        X: array-like, shape = [n_examples, n_features]
        n_examples is the number of training examples, n_features is number of features
        y: array-like, shape = [n_examples]
        goal variable

        returns self
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter): # repeat
            errors = 0
            for xi, target in zip(X,y): # for each training data add weight
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update!=0.0)

            # store each iteration's errors
            self.errors_.append(errors)
        return self

    def net_input(self,X):
        """
        Calculate net out
        """
        return np.dot(X,self.w_[1:]) + self.w_[0]

    def predict(self,X):
        """
        After one step return class label
        """
        return np.where(self.net_input(X) >= 0.0,1,-1)

if __name__ == "__main__":
    s = os.path.join('https://archive.ics.uci.edu','ml','machine-learning-databases','iris','iris.data')
    df = pd.read_csv(s,header=None,encoding='utf-8')

    y = df.iloc[0:100,4].values
    y = np.where(y=='Iris-setosa',-1,1)
    X = df.iloc[0:100,[0,2]].values

    ppn = Perceptron(eta=0.1,n_iter=10)
    ppn.fit(X,y)

    plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of update')
    plt.show()

    # plot decision boundaries
    plot_decision_regions(X,y,classifier=ppn)
    # set axis labels
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    # legend
    plt.legend(loc='upper left')

    plt.show()