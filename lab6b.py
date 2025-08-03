import numpy as np

class RecurrentNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.1, time_steps=5):
        self.input_dim= input_dim
        self.hidden_dim= hidden_dim
        self.output_dim= output_dim
        self.learning_rate= learning_rate
        self.time_steps= time_steps

        self.Wxh= np.random.rand(hidden_dim, input_dim)* 0.01
        self.Whh= np.random.randn(hidden_dim, hidden_dim)*0.01
        self.Why= np.random.randn(output_dim, hidden_dim)* 0.01
        self.by= np.zeros((output_dim,1))
        self.bh= np.zeros((hidden_dim, 1))
    
    def forward(self, inputs):
        h= np.zeros((self.hidden_dim,1)) #initialize hidden state
        self.h_states={-1:h} #hidden state h at time step -1
        self.output={}

        for t in range(self.time_steps):
            h= np.tanh(np.dot(self.Wxh, inputs[t])+ np.dot(self.Whh, self.h_states[t-1])+ self.bh)
            y= np.dot(self.Why, h)+ self.by
            self.h_states[t]=h
            self.output[t]= y
        return self.output
    
    def backward(self, inputs, target):
        dWxh, dWhh, dWhy= np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby= np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next= np.zeros((self.hidden_dim,1))

        for t in reversed(range(self.time_steps)):
            dy= self.output[t]- target[t]
            dWhy += np.dot(dy, self.h_states[t].T)
            dby += dy

            dh= np.dot(self.Why.T, dy) + dh_next
            dh_raw= (1-(self.h_states[t])**2) *dh
            dbh += dh_raw

            dWhh += np.dot(dh_raw, self.h_states[t-1].T)
            dWxh += np.dot(dh_raw, inputs[t].T)
            dh_next += np.dot(self.Whh.T, dh_raw)
        
        for dparam in [dWxh, dWhh, dWhy, dby, dbh]:
            np.clip(dparam, -5,5,out=dparam)
        
        for param, dparam in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],[dWxh, dWhh, dWhy, dbh, dby]):
            param -= self.learning_rate* dparam
    
    def train(self, data, labels, epochs=100):
        for epoch in range(epochs):
            loss=0
            for inputs, targets in zip(data, labels):
                output= self.forward(inputs)
                self.backward(inputs, targets)
                loss += np.sum((output[self.time_steps-1]- targets[self.time_steps-1])**2)/2
            if epoch%10 ==0:
                print(f"For epoch {epoch} loss is : {loss} ")

if __name__== "__main__":
    time_steps=5
    input_dim=3
    output_dim=2
    hidden_dim=4
    data= [np.random.randn(time_steps, input_dim,1)* 0.1 for _ in range(100)]
    labels= [np.random.randn(time_steps, output_dim,1)* 0.1 for _ in range(100)]

    rnn= RecurrentNeuralNetwork(input_dim, hidden_dim, output_dim, learning_rate=0.01)
    rnn.train(data, labels, epochs=50)
