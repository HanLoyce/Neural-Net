from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        if initialize_method == np.random.normal:
            std = np.sqrt(2.0 / in_dim)
            self.W = np.random.normal(loc=0.0, scale=std, size=(in_dim, out_dim))
        else:
            self.W = initialize_method(size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        output = X @ self.W + self.b
        return output

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = (grad.T @ self.input).T + self.weight_decay_lambda * self.W if self.weight_decay else (grad.T @ self.input).T
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) 
        return grad @ self.W.T

    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # W: [out_channels, in_channels, k, k], b: [out_channels]
        fan_in = in_channels * kernel_size * kernel_size
        if initialize_method == np.random.normal:
            std = np.sqrt(2.0 / fan_in)
            self.W = np.random.normal(loc=0.0, scale=std, size=(out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros((out_channels,))

        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.cache = None

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        self.input = X
        batch_size, in_channels, H, W = X.shape
        assert in_channels == self.in_channels

        k = self.kernel_size
        s = self.stride
        p = self.padding

        Ho = (H + 2 * p - k) // s + 1
        Wo = (W + 2 * p - k) // s + 1

        x_pad = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        # im2col: [N*Ho*Wo, C*k*k]
        x_col = np.zeros((batch_size * Ho * Wo, in_channels * k * k), dtype=X.dtype)
        row = 0
        for i in range(Ho):
            h_start = i * s
            h_end = h_start + k
            for j in range(Wo):
                w_start = j * s
                w_end = w_start + k
                patch = x_pad[:, :, h_start:h_end, w_start:w_end].reshape(batch_size, -1)
                x_col[row:row + batch_size] = patch
                row += batch_size

        w_col = self.W.reshape(self.out_channels, -1)
        out_col = x_col @ w_col.T + self.b
        output = out_col.reshape(Ho, Wo, batch_size, self.out_channels).transpose(2, 3, 0, 1)

        self.cache = (X.shape, x_pad, x_col, Ho, Wo)
        return output
    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        X_shape, x_pad, x_col, Ho, Wo = self.cache
        batch_size, in_channels, H, W = X_shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        dout_col = grads.transpose(2, 3, 0, 1).reshape(-1, self.out_channels)
        w_col = self.W.reshape(self.out_channels, -1)

        db = np.sum(dout_col, axis=0)
        dW = (dout_col.T @ x_col).reshape(self.W.shape)
        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W

        dx_col = dout_col @ w_col
        dx_pad = np.zeros_like(x_pad)

        row = 0
        for i in range(Ho):
            h_start = i * s
            h_end = h_start + k
            for j in range(Wo):
                w_start = j * s
                w_end = w_start + k
                patch_grad = dx_col[row:row + batch_size].reshape(batch_size, in_channels, k, k)
                dx_pad[:, :, h_start:h_end, w_start:w_end] += patch_grad
                row += batch_size

        if p > 0:
            dx = dx_pad[:, :, p:-p, p:-p]
        else:
            dx = dx_pad

        self.grads['W'] = dW
        self.grads['b'] = db
        return dx
        
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output


class Flatten(Layer):
    """
    Flatten tensor [N, C, H, W] to [N, C*H*W].
    """
    def __init__(self) -> None:
        super().__init__()
        self.optimizable = False
        self.input_shape = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.labels = None
        self.output = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        self.labels = labels
        if self.has_softmax:
            self.output = softmax(predicts)
        else:
            self.output = predicts
        loss = -np.sum(np.log(self.output[np.arange(len(self.labels)),self.labels] + 1e-15)) / len(self.labels)
        return loss

        
    def backward(self):
        
        # first compute the grads from the loss to the input
        self.grads = self.output.copy()
        self.grads[np.arange(len(self.output)),self.labels] -= 1
        self.grads /= len(self.labels)
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition