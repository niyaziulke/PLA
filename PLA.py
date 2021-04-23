import numpy as np
import matplotlib.pyplot as plt

def sign(x):
    """ Returns sign of a number"""
    return abs(x) / x

def create_numbers(N, dim=2, scope = 400):
    X = np.random.rand(N,dim+1)*scope-scope/2
    X[:,0] = 1 # x0 values == 1 (bias)
    f = lambda x,y: sign(3*x+y-1) 
    labels = f(X[:,1],X[:,2])
    return X,labels


def PLA(X,c):
    w = np.random.rand(3)
    incomplete = True
    while incomplete:
        wX = np.dot(X,w)
        classify = sign(c * wX)
        missclass_index = np.where(classify == -1)
        if len(missclass_index[0]):
            update = missclass_index[0][0]
            w = w + c[update] * X[update,:]
        else:
            incomplete = False
    return w

def plot_graph(X, labels, w, scope = 400):
    color_dict = {-1:"red", 1:"blue"}
    y_w = lambda x: -x*w[1]/w[2] - w[0]/w[2]
    y_formula = lambda x: -3*x + 1
    plt.scatter(x = X[:,1], y = X[:,2], c = [color_dict[label] for label in labels] ,s=0.2)
    plt.plot([-scope/2, scope/2],[y_w(-scope/2),y_w(scope/2)],c="purple")
    plt.plot([-scope/2, scope/2],[y_formula(-scope/2),y_formula(scope/2)],c="green")
    plt.xlim([-scope/2, scope/2])
    plt.ylim([-scope/2, scope/2])
    
def main(): 
    X,labels = create_numbers(5000)
    w_final = PLA(X,labels)
    plot_graph(X,labels,w_final)

main()