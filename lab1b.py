import numpy as np
from collections import Counter

class Neuron:
    def __init__(self,k):
        self.k=k
        self.X_train= None
        self.y_train= None

    def train(self, X_train, y_train):
        self.X_train= X_train
        self.y_train= y_train
    
    def predict(self, X_test):
        predictions= [self._predict(x) for x in X_test]
        return np.array(predictions)
    
    def distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def _predict(self,x):
        distances= [self.distance(x, x_train) for x_train in self.X_train]
        k_indices= np.argsort(distances)[:self.k]
        k_nearest_labels= [self.y_train[i] for i in k_indices]
        most_common= Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
if __name__=="__main__":
    k=3
    neuron= Neuron(k)
    X_train= np.array([[1,2],[2,3],[3,4],[4,5]])
    y_train= np.array([0,0,1,1])
    x_test= np.array([[5,6],[0,1]])
    neuron.train(X_train, y_train)
    predictions= neuron.predict(x_test)
    print(predictions)
        


              
