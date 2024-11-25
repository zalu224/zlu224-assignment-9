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

        #store activations and gradients for visualization
        self.activation_features = None
        self.gradients = {}

    def activation(self, x):
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
            sx = 1 / (1 + np.exp(-x))
            return sx * (1 - sx)
        
    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        #first layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)

        #second layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.out = np.tanh(self.z2)
        # TODO: store activations for visualization
        self.hidden_features = self.a1

        # out = ...
        return self.out

    def backward(self, X, y):
        m = X.shape[0]
        # TODO: compute gradients using chain rule
        # Output layer gradients
        delta2 = (self.out - y) * (1 - self.out**2)  # derivative of tanh
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)

        # TODO: update weights with gradient descent
         # Hidden layer gradients
        delta1 = np.dot(delta2, self.W2.T) * self.activation_derivative(self.z1)
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)

        # TODO: store gradients for visualization
        # Store gradients for visualization
        self.gradients = {
            'W1': dW1,
            'W2': dW2,
            'b1': db1,
            'b2': db2
        }
        
        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        # pass

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

    # TODO: Hyperplane visualization in the hidden space
    xx, yy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    zz = -(mlp.W2[0] * xx + mlp.W2[1] * yy + mlp.b2) / (mlp.W2[2] + 1e-10)
    ax_hidden.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    
    # Set hidden space view angle to match example
    ax_hidden.view_init(elev=20, azim=-45)

    # TODO: Distorted input space transformed by the hidden layer

    # TODO: Plot input layer decision boundary
    x_min, x_max = -3, 3
    y_min, y_max = -2, 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    Z = np.c_[xx.ravel(), yy.ravel()]
    Z_pred = mlp.forward(Z)
    Z_pred = Z_pred.reshape(xx.shape)
    
    ax_input.contourf(xx, yy, Z_pred, levels=20, cmap='bwr', alpha=0.3)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='black')
    
    # Set input space limits to match example
    ax_input.set_xlim([x_min, x_max])
    ax_input.set_ylim([y_min, y_max])

    # TODO: Visualize features and gradients as circles and edges 
    # The edge thickness visually represents the magnitude of the gradient
    pos = {
        'x1': (0.0, 0.0),
        'x2': (0.0, 1.0),
        'h1': (0.5, 0.0),
        'h2': (0.5, 0.5),
        'h3': (0.5, 1.0),
        'y': (1.0, 0.5)
    }
    
    # Draw nodes
    for name, (x, y) in pos.items():
        circle = Circle((x, y), 0.08, color='blue', fill=True)
        ax_gradient.add_patch(circle)
        ax_gradient.text(x-0.03, y-0.03, name, color='white', fontweight='bold')
    
    
    # Draw edges with gradient-based thickness
    max_thickness = 2
    min_thickness = 0.5

    for i in range(2):
        for j in range(3):
            # Input to hidden connections
            start = pos[f'x{i+1}']
            end = pos[f'h{j+1}']
            weight = abs(mlp.gradients['W1'][i, j])
            thickness = max_thickness * weight / np.max(np.abs(mlp.gradients['W1']))
            ax_gradient.plot([start[0], end[0]], [start[1], end[1]], 
                           'purple', linewidth=thickness, alpha=0.6)
    
    # Hidden to output connections
    for i in range(3):
        start = pos[f'h{i+1}']
        end = pos['y']
        weight = abs(mlp.gradients['W2'][i, 0])
        thickness = min_thickness + (max_thickness - min_thickness) * weight / (np.max(np.abs(mlp.gradients['W2'])) + 1e-10)
        ax_gradient.plot([start[0], end[0]], [start[1], end[1]], 
                        color='purple', linewidth=thickness, alpha=0.6)
    
    # Set titles and labels
    ax_hidden.set_title(f'Hidden Space at Step {frame*10}')
    ax_input.set_title(f'Input Space at Step {frame*10}')
    ax_gradient.set_title(f'Gradients at Step {frame*10}')
    
    # Set gradient view limits
    ax_gradient.set_xlim([-0.1, 1.1])
    ax_gradient.set_ylim([-0.1, 1.1])
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