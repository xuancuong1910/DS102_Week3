import numpy as np

class SVM:
    def __init__(self, C: float = 1.0, lr: float = 0.001, epochs: int = 100):
        self.C = C              
        self.lr = lr            
        self.epochs = epochs    
        self.w = None           
        self.b = 0              

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.epochs):
            
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            for idx in indices:
                x_i = X[idx]
                y_i = y[idx]
                condition = y_i * (np.dot(x_i, self.w) + self.b)
                if condition >= 1:
                   
                    delta_w = self.w
                    self.w = self.w - self.lr * delta_w
                else:
                    
                    delta_w = self.w - self.C * y_i * x_i
                    self.w = self.w - self.lr * delta_w
                    self.b = self.b + self.lr * self.C * y_i

    def predict(self, X: np.ndarray):
       
        y_hat = np.dot(X, self.w) + self.b
        return np.sign(y_hat)

    def hinge_loss(self, y: np.ndarray, y_hat: np.ndarray):
        
        delta = 1 - y * y_hat
       
        hinge_loss_val = np.maximum(0, delta)
        return 0.5 * np.dot(self.w, self.w) + self.C * np.sum(hinge_loss_val)
