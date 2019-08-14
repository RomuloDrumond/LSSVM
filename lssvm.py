import numpy as np
from numpy import dot, exp
from scipy.spatial.distance import cdist

class LSSVM:
    'Class that implements the Least-Squares Support Vector Machine.'
    
    def __init__(self, gamma=1, kernel='rbf', **kernel_params): 
        self.gamma = gamma
        
        self.x        = None
        self.y        = None
        self.y_labels = None
        
        # model params
        self.alpha = None
        self.b     = None
        
        self.kernel = LSSVM.get_kernel(kernel, **kernel_params)
        
           
    @staticmethod
    def get_kernel(name, **params):
        
        def linear(x_i, x_j):                           
            return dot(x_i, x_j.T)
        
        def poly(x_i, x_j, d=params.get('d',3)):        
            return ( dot(x_i, x_j.T) + 1 )**d
        
        def rbf(x_i, x_j, sigma=params.get('sigma',1)):
            if x_i.ndim==x_i.ndim and x_i.ndim==2: # both matrices
                return exp( -cdist(x_i,x_j)**2 / sigma**2 )
            
            else: # both vectors or a vector and a matrix
                return exp( -( dot(x_i,x_i.T) + dot(x_j,x_j.T)- 2*dot(x_i,x_j) ) / sigma**2 )
#             temp = x_i.T - X
#             return exp( -dot(temp.temp) / sigma**2 )
                
        kernels = {'linear': linear, 'poly': poly, 'rbf': rbf}
                
        if kernels.get(name) is None: 
            raise KeyError("Kernel '{}' is not defined, try one in the list: {}.".format(
                name, list(kernels.keys())))
        else: return kernels[name]
        
    
    def opt_params(self, X, y_values):
        sigma = np.multiply( y_values*y_values.T, self.kernel(X,X) )

        A_cross = np.linalg.pinv(np.block([
            [0,                           y_values.T                   ],
            [y_values,   sigma + self.gamma**-1 * np.eye(len(y_values))]
        ]))

        B = np.array([0]+[1]*len(y_values))

        solution = dot(A_cross, B)
        b     = solution[0]
        alpha = solution[1:]
        
        return (b, alpha)
            
    
    def fit(self, X, Y, verboses=0):
        self.x = X
        self.y = Y
        self.y_labels = np.unique(Y, axis=0)
        
        if len(self.y_labels)==2: # binary classification
            # converting to -1/+1
            y_values = np.where(
                (Y == self.y_labels[0]).all(axis=1)
                ,-1,+1)[:,np.newaxis] # making it a column vector
            
            self.b, self.alpha = self.opt_params(X, y_values)
        
        else: # multiclass classification
              # ONE-VS-ALL APPROACH
            n_classes = len(self.y_labels)
            self.b     = np.zeros(n_classes)
            self.alpha = np.zeros((n_classes, len(Y)))
            for i in range(n_classes):
                # converting to +1 for the desired class and -1 for all other classes
                y_values = np.where(
                    (Y == self.y_labels[i]).all(axis=1)
                    ,+1,-1)[:,np.newaxis] # making it a column vector
  
                self.b[i], self.alpha[i] = self.opt_params(X, y_values)

        
    def predict(self, X):
        K = self.kernel(self.x, X)
        
        if len(self.y_labels)==2: # binary classification
            y_values = np.where(
                (self.y == self.y_labels[0]).all(axis=1),
                -1,+1)[:,np.newaxis] # making it a column vector

            Y = np.sign( dot( np.multiply(self.alpha, y_values.flatten()), K ) + self.b)
            
            y_pred_labels = np.where(Y==-1, self.y_labels[0], 
                                     self.y_labels[1])
        
        else: # multiclass classification, ONE-VS-ALL APPROACH
            Y = np.zeros((len(self.y_labels), len(X)))
            for i in range(len(self.y_labels)):
                y_values = np.where(
                    (self.y == self.y_labels[i]).all(axis=1),
                    +1, -1)[:,np.newaxis] # making it a column vector
                Y[i] = dot( np.multiply(self.alpha[i], y_values.flatten()), K ) + self.b[i] # no sign function applied
            
            predictions = np.argmax(Y, axis=0)
            y_pred_labels = np.array([self.y_labels[i] for i in predictions])
            
        return y_pred_labels
    
    

    
######################################################################################################################


import torch

class LSSVM_GPU:
    'Class that implements the Least-Squares Support Vector Machine on GPU.'
    
    def __init__(self, gamma=1, kernel='rbf', **kernel_params): 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.gamma = gamma
        
        self.x        = None
        self.y        = None
        self.y_labels = None
        
        # model params
        self.alpha = None
        self.b     = None
        
        self.kernel = LSSVM_GPU.get_kernel(kernel, **kernel_params) # saving kernel function
        
           
    @staticmethod
    def get_kernel(name, **params):
        
        def linear(x_i, x_j):                           
            return torch.mm(x_i, torch.t(x_j))
        
        def poly(x_i, x_j, d=params.get('d',3)):        
            return ( torch.mm(x_i, torch.t(x_j)) + 1 )**d
        
        def rbf(x_i, x_j, sigma=params.get('sigma',1)):
            if x_i.ndim==x_i.ndim and x_i.ndim==2: # both matrices
                return torch.exp( -torch.cdist(x_i,x_j)**2 / sigma**2 )
            
            else: # both vectors or a vector and a matrix
                return torch.exp( -( torch.dot(x_i,torch.t(x_i)) + torch.dot(x_j,torch.t(x_j))- 2*torch.dot(x_i,x_j) ) 
                                 / sigma**2 )
#             temp = x_i.T - X
#             return exp( -dot(temp.temp) / sigma**2 )
                
        kernels = {'linear': linear, 'poly': poly, 'rbf': rbf}
                
        if kernels.get(name) is None: 
            raise KeyError("Kernel '{}' is not defined, try one in the list: {}.".format(
                name, list(kernels.keys())))
        else: return kernels[name]
        
    
    def opt_params(self, X, y_values):
        sigma = ( torch.mm(y_values, torch.t(y_values)) ) * self.kernel(X,X)

        A_cross = torch.pinverse(torch.cat(( 
            # block matrix
            torch.cat(( torch.tensor(0, dtype=X.dtype, device=self.device).view(1,1),
                        torch.t(y_values)
                      ),dim=1),
            torch.cat(( y_values, 
                        sigma + self.gamma**-1 * torch.eye(len(y_values), dtype=X.dtype, device=self.device) 
                      ),dim=1)
        ),dim=0))

        B = torch.tensor([0]+[1]*len(y_values), dtype=X.dtype, device=self.device).view(-1,1)

        solution = torch.mm(A_cross, B)
        b     = solution[0]
        alpha = solution[1:].view(-1) # 1D array form
        
        return (b, alpha)
            
    
    def fit(self, X, Y, verboses=0):
        # converting to tensors and passing to GPU
        X = torch.from_numpy(X).to(self.device)
        Y = torch.from_numpy(Y).to(self.device)
        self.x = X
        self.y = Y
        self.y_labels = torch.unique(Y, dim=0)
        
        if len(self.y_labels)==2: # binary classification
            # converting to -1/+1
            y_values = torch.where(
                (Y == self.y_labels[0]).all(axis=1)
                ,torch.tensor(-1, dtype=X.dtype, device=self.device)
                ,torch.tensor(+1, dtype=X.dtype, device=self.device)
            ).view(-1,1) # making it a column vector
            
            self.b, self.alpha = self.opt_params(X, y_values)
        
        else: # multiclass classification
              # ONE-VS-ALL APPROACH
            n_classes = len(self.y_labels)
            self.b     = torch.empty(n_classes,         dtype=X.dtype, device=self.device)
            self.alpha = torch.empty(n_classes, len(Y), dtype=X.dtype, device=self.device)
            for i in range(n_classes):
                # converting to +1 for the desired class and -1 for all other classes
                y_values = torch.where(
                    (Y == self.y_labels[i]).all(axis=1)
                    ,torch.tensor(+1, dtype=X.dtype, device=self.device)
                    ,torch.tensor(-1, dtype=X.dtype, device=self.device)
                ).view(-1,1) # making it a column vector
                  
                self.b[i], self.alpha[i] = self.opt_params(X, y_values)

        
    def predict(self, X):
        X = torch.from_numpy(X).to(self.device)
        K = self.kernel(self.x, X)
        
        if len(self.y_labels)==2: # binary classification
            y_values = torch.where(
                (self.y == self.y_labels[0]).all(axis=1)
                ,torch.tensor(-1, dtype=X.dtype, device=self.device)
                ,torch.tensor(+1, dtype=X.dtype, device=self.device)
            )
            
            Y = torch.sign( torch.mm( (self.alpha*y_values).view(1,-1), K ) + self.b)
            
            y_pred_labels = torch.where(Y==-1,        self.y_labels[0],
                                        self.y_labels[1]
                                       ).view(-1) # convert to flat array
        
        else: # multiclass classification, ONE-VS-ALL APPROACH
            Y = torch.empty((len(self.y_labels), len(X)), dtype=X.dtype, device=self.device)
            for i in range(len(self.y_labels)):
                y_values = torch.where(
                    (self.y == self.y_labels[i]).all(axis=1)
                    ,torch.tensor(+1, dtype=X.dtype, device=self.device)
                    ,torch.tensor(-1, dtype=X.dtype, device=self.device)
                )

                Y[i] = torch.mm( (self.alpha[i]*y_values).view(1,-1), K ) + self.b[i] # no sign function applied
            
            predictions = torch.argmax(Y, axis=0)
            y_pred_labels = torch.stack([self.y_labels[i] for i in predictions])
            
        return y_pred_labels