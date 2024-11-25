import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function

        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
        
        # For storing activations and gradients
        self.hidden_features = None
        self.gradients = None

    def activation(self, x): # based on the readme recommendations
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
            
    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation_fn == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        
    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # TODO: store activations for visualization
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        out = np.tanh(self.z2)

        self.hidden_features = self.a1
        return out

    def backward(self, X, y):
         
        m = X.shape[0]
        
        # TODO: compute gradients using chain rule
         # Chain rule
        delta2 = (self.forward(X) - y) * (1 - self.forward(X)**2)
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m
        
        # TODO: update weights with gradient descent

        delta1 = np.dot(delta2, self.W2.T) * self.activation_derivative(self.z1)
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m

        # TODO: store gradients for visualization
        # Weight update 
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.gradients = {
            'W1': dW1,
            'W2': dW2,
            'b1': db1,
            'b2': db2
        }
        
def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # TODO: Plot hidden features
    hidden_features = mlp.hidden_features
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_xlim([-1, 1])
    ax_hidden.set_ylim([-1, 1])
    ax_hidden.set_zlim([-1, 1])

    # TODO: Hyperplane visualization in the hidden space
    xx = np.linspace(-1, 1, 20)
    yy = np.linspace(-1, 1, 20)
    zz = np.linspace(-1, 1, 20)
    XX, YY, ZZ = np.meshgrid(xx, yy, zz)
    hidden_points = np.column_stack((XX.ravel(), YY.ravel(), ZZ.ravel()))
    hyperplane = np.dot(hidden_points, mlp.W2) + mlp.b2


    hyperplane = hyperplane.reshape(XX.shape)
    for i in range(XX.shape[2]):
        mask = np.abs(hyperplane[:,:,i]) < 0.1
        if mask.any():
            ax_hidden.scatter(XX[:,:,i][mask], YY[:,:,i][mask], ZZ[:,:,i][mask], 
                            color='lightgray', alpha=0.2, s=1)
            
    # TODO: Distorted input space transformed by the hidden layer
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20),
                        np.linspace(y_min, y_max, 20))
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    transformed = mlp.activation(np.dot(grid_points, mlp.W1) + mlp.b1)
    ax_hidden.scatter(transformed[:,0], transformed[:,1], transformed[:,2], 
                     alpha=0.1, color='gray')

    # TODO: Plot input layer decision boundary
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    Z = mlp.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax_input.contourf(xx, yy, Z, cmap='RdBu', alpha=0.3)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr')

    # TODO: Visualize features and gradients as circles and edges 
    # The edge thickness visually represents the magnitude of the gradient
    nodes_pos = {
        'x1': (0.0, 0.0),
        'x2': (0.0, 1.0),
        'h1': (0.4, 0.2),
        'h2': (0.4, 0.5),
        'h3': (0.4, 0.8),
        'y': (0.8, 0.5)
    }
    for name, pos in nodes_pos.items():
        circle = Circle(pos, 0.05, color='blue', fill=True)
        ax_gradient.add_patch(circle)
        ax_gradient.text(pos[0]-0.02, pos[1]-0.02, name)
    max_grad = max(np.abs(mlp.gradients['W1']).max(), np.abs(mlp.gradients['W2']).max())
    
    # input to hidden connections
    for i in range(2):
        for j in range(3):
            weight = abs(mlp.gradients['W1'][i, j]) / max_grad
            ax_gradient.plot([nodes_pos[f'x{i+1}'][0], nodes_pos[f'h{j+1}'][0]],
                           [nodes_pos[f'x{i+1}'][1], nodes_pos[f'h{j+1}'][1]],
                           'purple', linewidth=weight*3)
    
    # hidden to output connections
    for i in range(3):
        weight = abs(mlp.gradients['W2'][i, 0]) / max_grad
        ax_gradient.plot([nodes_pos[f'h{i+1}'][0], nodes_pos['y'][0]],
                        [nodes_pos[f'h{i+1}'][1], nodes_pos['y'][1]],
                        'purple', linewidth=weight*3)
    
    ax_gradient.set_xlim([-0.2, 1.0])
    ax_gradient.set_ylim([-0.2, 1.2])
    ax_gradient.axis('equal')
    ax_gradient.axis('off')


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)