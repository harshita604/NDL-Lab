import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivative_sigmoid(x):
    return x*(1-x)


inputs= np.array([[0,0],[0,1],[1,0],[1,1]])
outputs= np.array([[0],[1],[1],[0]])
np.random.seed(0)

epochs=1000
l_r=[0.01,0.1,0.5]
for lr in l_r:
    hidden_weights= np.random.uniform(-1,1,(2,2))
    hidden_bias= np.random.uniform(-1,1,(1,2))
    output_weights= np.random.uniform(-1,1,(2,1))
    output_bias= np.random.uniform(-1,1,(1,1))
    print(f"Training with learning rate: {lr}")
    for epoch in range(epochs):

        hidden_layer_input= np.dot(inputs, hidden_weights) + hidden_bias
        hidden_layer_output= sigmoid(hidden_layer_input)
        output_layer_input= np.dot(hidden_layer_output, output_weights)+ output_bias
        predicted_output= sigmoid(output_layer_input)

        error= outputs- predicted_output
        mse= np.mean(np.square(error))

        d_predicted_output= error* derivative_sigmoid(predicted_output)
        d_hidden_layer= np.dot(d_predicted_output, output_weights.T)* derivative_sigmoid(hidden_layer_output)

        output_weights += np.dot(hidden_layer_output.T, d_predicted_output)* lr
        output_bias += np.sum(d_predicted_output, axis=0, keepdims=True)* lr
        hidden_weights += np.dot(inputs.T, d_hidden_layer)*lr
        hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True)*lr

        if mse<0.01:
            print(f"Mean squared error: {mse}, early stopping at {epoch+1}")
            break
    print(f"MSE: {mse}")

    print("Results for XOR Gate:\n")
    for i in range(len(inputs)):
        hidden_layer_input= np.dot(inputs[i], hidden_weights)+ hidden_bias
        hidden_layer_output= sigmoid(hidden_layer_input)
        output_layer_input= np.dot(hidden_layer_output, output_weights)+ output_bias
        predicted_output= sigmoid(output_layer_input)
        print(f"Inputs: {inputs[i]}, Output: {np.round(predicted_output[0],2)}")

