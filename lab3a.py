w1, w2,b= 0.5,0.5,1
def activate(x):
    return 1 if x>=0 else 0

def train_perceptron(input, desired_output, epochs, l_r):
    global w1,w2,b
    for epoch in range(epochs):
        total_error=0
        for i in range(len(input)):
            A,B=input[i]
            target_output= desired_output[i]
            output= activate(w1*A + w2*B +b)
            error= target_output-output  #imp
            w1+= error*l_r*A
            w2+= error*l_r*B
            b+=error*l_r
            total_error+= abs(error)
        if total_error==0:
            break

input=[(0,0),(0,1),(1,0),(1,1)]
output=[0,0,0,1]
epochs=1000
l_r=0.1
train_perceptron(input, output, epochs, l_r)

for i in range(len(input)):
    A,B= input[i]
    output= activate(w1*A + w2*B+ b)
    print(A,B, ":", output)

# AND, OR: 0.5, 0.5, 1
#NAND: -1,-1,1
#NOR : -1,-1,0