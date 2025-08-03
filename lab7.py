import numpy as np

def train_hopfield(patterns):
    num_neurons= len(patterns[0])
    weight_matrix= np.zeros((num_neurons, num_neurons))

    for pattern in patterns:
        pattern= np.array(pattern).reshape(-1,1)
        weight_matrix += pattern @ pattern.T
    np.fill_diagonal(weight_matrix, 0)
    return weight_matrix

def recall_pattern(weight_matrix, input_pattern, max_iter):
    output_pattern= np.array(input_pattern)
    for _ in range(max_iter):
        for i in range (len(output_pattern)):
            result= np.dot(weight_matrix[i], output_pattern)
            output_pattern[i]=1 if result>=0 else -1
    return output_pattern

original= [-1,1,1,1,-1,-1,-1,1]
noisy= [1,1,1,1,-1,-1,1,1]

weight_matrix= train_hopfield([original])
x= recall_pattern(weight_matrix, noisy, max_iter=10)
print(x)