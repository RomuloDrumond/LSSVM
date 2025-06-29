import numpy as np
from numpy import dot
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

from utils.kernel import get_kernel
from utils.import_export import dump_model, load_model
from utils.conversion import numpy_json_encoder


class LSSVC(BaseEstimator, ClassifierMixin):
    """A class that implements the Least Squares Support Vector Machine 
    for classification tasks.

    It uses Numpy pseudo-inverse function to solve the dual optimization 
    problem with ordinary least squares. In multiclass classification 
    problems the approach used is one-vs-all, so, a model is fit for each 
    class while considering the others a single set of the same class.
    
    # Parameters:
    - gamma: float, default = 1.0
        Constant that control the regularization of the model, it may vary 
        in the set (0, +infinity). The closer gamma is to zero, the more 
        regularized the model will be.
    - kernel: {'linear', 'poly', 'rbf'}, default = 'rbf'
        The kernel used for the model, if set to 'linear' the model 
        will not take advantage of the kernel trick, and the LSSVC maybe only
        useful for linearly separable problems.
    - kernel_params: **kwargs, default = depends on 'kernel' choice
        If kernel = 'linear', these parameters are ignored. If kernel = 'poly',
        'd' is accepted to set the degree of the polynomial, with default = 3. 
        If kernel = 'rbf', 'sigma' is accepted to set the radius of the 
        gaussian function, with default = 1. 
     
    # Attributes:
    - All hyperparameters of section "Parameters".
    - alpha: ndarray of shape (1, n_support_vectors) if in binary 
             classification and (n_classes, n_support_vectors) for 
             multiclass problems
        Each column is the optimum value of the dual variable for each model
        (using the one-vs-all approach we have n_classes == n_classifiers), 
        it can be seen as the weight given to the support vectors 
        (sv_x, sv_y). As usually there is no alpha == 0, we have 
        n_support_vectors == n_train_samples.
    - b: ndarray of shape (1,) if in binary classification and (n_classes,) 
         for multiclass problems 
        The optimum value of the bias of the model.
    - sv_x: ndarray of shape (n_support_vectors, n_features)
        The set of the supporting vectors attributes, it has the shape 
        of the training data.
    - sv_y: ndarray of shape (n_support_vectors, n)
        The set of the supporting vectors labels. If the label is represented 
        by an array of n elements, the sv_y attribute will have n columns.
    - y_labels: ndarray of shape (n_classes, n)
        The set of unique labels. If the label is represented by an array 
        of n elements, the y_label attribute will have n columns.
    - K: function, default = rbf()
        Kernel function.
    """
    
    def __init__(self, gamma=1, kernel='rbf', **kernel_params): 
        # Hyperparameters
        self.gamma = gamma
        self.kernel = kernel
        self.kernel_params = kernel_params
        
        # Model parameters
        self.alpha = None
        self.b = None
        self.sv_x = None
        self.sv_y = None
        self.y_labels = None
        
        self.K = get_kernel(kernel, **kernel_params)
    
    def _optimize_parameters(self, X, y_values):
        """Help function that optimizes the dual variables through the 
        use of the kernel matrix pseudo-inverse.
        """
        sigma = np.multiply(y_values*y_values.T, self.K(X,X))
        
        A = np.block([
            [0, y_values.T],
            [y_values, sigma + self.gamma**-1 * np.eye(len(y_values))]
        ])
        B = np.array([0]+[1]*len(y_values))
        
        A_cross = np.linalg.pinv(A)

        solution = dot(A_cross, B)
        b = solution[0]
        alpha = solution[1:]
        
        return (b, alpha)
    
    def fit(self, X, y):
        """Fits the model given the set of X attribute vectors and y labels.
        - X: ndarray of shape (n_samples, n_attributes)
        - y: ndarray of shape (n_samples,) or (n_samples, n)
            If the label is represented by an array of n elements, the y 
            parameter must have n columns.
        """
        y_reshaped = y.reshape(-1,1) if y.ndim==1 else y

        self.sv_x = X
        self.sv_y = y_reshaped
        self.y_labels = np.unique(y_reshaped, axis=0)
        
        if len(self.y_labels) == 2: # binary classification
            # converting to -1/+1
            y_values = np.where(
                (y_reshaped == self.y_labels[0]).all(axis=1)
                ,-1,+1)[:,np.newaxis] # making it a column vector
            
            self.b, self.alpha = self._optimize_parameters(X, y_values)
        
        else: # multiclass classification, one-vs-all approach
            n_classes = len(self.y_labels)
            self.b = np.zeros(n_classes)
            self.alpha = np.zeros((n_classes, len(y_reshaped)))
            
            for i in range(n_classes):
                # converting to +1 for the desired class and -1 for all 
                # other classes
                y_values = np.where(
                    (y_reshaped == self.y_labels[i]).all(axis=1)
                    ,+1,-1)[:,np.newaxis]
  
                self.b[i], self.alpha[i] = self._optimize_parameters(X, y_values)
        
    def predict(self, X):
        """Predicts the labels of data X given a trained model.
        - X: ndarray of shape (n_samples, n_attributes)
        """
        if self.alpha is None:
            raise Exception(
                "The model doesn't see to be fitted, try running .fit() method first"
            )

        X_reshaped = X.reshape(1,-1) if X.ndim==1 else X
        KxX = self.K(self.sv_x, X_reshaped)
        
        if len(self.y_labels)==2: # binary classification
            y_values = np.where(
                (self.sv_y == self.y_labels[0]).all(axis=1),
                -1,+1)[:,np.newaxis]

            y = np.sign(dot(np.multiply(self.alpha, y_values.flatten()), KxX) + self.b)
            
            y_pred_labels = np.where(y==-1, self.y_labels[0], self.y_labels[1])
        
        else: # multiclass classification, one-vs-all approach
            y = np.zeros((len(self.y_labels), len(X)))
            for i in range(len(self.y_labels)):
                y_values = np.where(
                    (self.sv_y == self.y_labels[i]).all(axis=1),
                    +1, -1)[:,np.newaxis]
                y[i] = dot(np.multiply(self.alpha[i], y_values.flatten()), KxX) + self.b[i]
            
            predictions = np.argmax(y, axis=0)
            y_pred_labels = np.array([self.y_labels[i] for i in predictions])
            
        return y_pred_labels

    def score(self, X, y):
        """Calculates the mean accuracy on the given test data and labels.
        - X: ndarray of shape (n_samples, n_attributes)
            Test samples.
        - y: ndarray of shape (n_samples,) or (n_samples, n)
            True labels for X.
        Returns:
        - score: float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)

        # Ensure y and y_pred are comparable for accuracy_score
        # accuracy_score expects 1D arrays. If y or y_pred are not,
        # and they represent multi-label or multi-output,
        # a simple flatten might not be appropriate.
        # However, for typical classification, if y is (n_samples, 1)
        # and y_pred is (n_samples, 1), they should be raveled.
        # If y_labels are multi-dimensional (e.g. one-hot encoded originally),
        # predict already converts them back to a single label per sample.
        # So, y needs to be in a compatible format.

        y_true_processed = y.ravel() if y.ndim > 1 and y.shape[1] == 1 else y
        y_pred_processed = y_pred.ravel() if y_pred.ndim > 1 and y_pred.shape[1] == 1 else y_pred

        # If y_labels had multiple columns (e.g. [0,1] vs [1,0]),
        # y_true_processed and y_pred_processed might need more complex handling.
        # For now, assuming y is comparable to the output of predict.
        # If y_true is multi-column and not just a column vector, direct comparison might fail.
        # This implementation assumes y is either 1D or a 2D column vector.
        if y_true_processed.ndim > 1 or y_pred_processed.ndim > 1:
            # This case might happen if y_labels are inherently multi-dimensional (e.g. [[0,1],[1,0]])
            # and y is passed in that format. accuracy_score won't work directly.
            # A more robust way would be to check equality row by row if shapes are compatible.
            if y_true_processed.shape == y_pred_processed.shape:
                 correct_predictions = np.all(y_true_processed == y_pred_processed, axis=1).sum()
                 return correct_predictions / len(y_true_processed)
            else:
                # This indicates a shape mismatch that needs specific handling
                # For now, raising an error or returning 0 might be options.
                # Or, try to make them compatible if a known transformation exists.
                # This part depends on the expected structure of y for complex labels.
                # Assuming for now predict output and y should be made 1D if they are column vectors.
                pass # Let accuracy_score handle it or raise error

        return accuracy_score(y_true_processed, y_pred_processed)

    def dump(self, filepath='model', only_hyperparams=False):
        """This method saves the model in a JSON format.
        - filepath: string, default = 'model'
            File path to save the model's json.
        - only_hyperparams: boolean, default = False
            To either save only the model's hyperparameters or not, it 
            only affects trained/fitted models.
        """
        model_json = {
            'type': 'LSSVC',
            'hyperparameters': {
                'gamma': self.gamma,
                'kernel': self.kernel,
                'kernel_params': self.kernel_params
            }           
        }

        if (self.alpha is not None) and (not only_hyperparams):
            model_json['parameters'] = {
                'alpha': self.alpha,
                'b': self.b,
                'sv_x': self.sv_x,
                'sv_y': self.sv_y,
                'y_labels': self.y_labels
            }
        
        dump_model(model_dict=model_json, file_encoder=numpy_json_encoder, filepath=filepath)
        
    @classmethod
    def load(cls, filepath, only_hyperparams=False):
        """This class method loads a model from a .json file.
        - filepath: string
            The model's .json file path.
        - only_hyperparams: boolean, default = False
            To either load only the model's hyperparameters or not, it 
            only has effects when the dump of the model as done with the
            model's parameters.
        """
        model_json = load_model(filepath=filepath)

        if model_json['type'] != 'LSSVC':
            raise Exception(
                f"Model type '{model_json['type']}' doesn't match 'LSSVC'"
            )

        lssvc = LSSVC(
            gamma = model_json['hyperparameters']['gamma'],
            kernel = model_json['hyperparameters']['kernel'],
            **model_json['hyperparameters']['kernel_params']
        )

        if (model_json.get('parameters') is not None) and (not only_hyperparams):
            lssvc.alpha = np.array(model_json['parameters']['alpha'])
            lssvc.b = np.array(model_json['parameters']['b'])
            lssvc.sv_x = np.array(model_json['parameters']['sv_x'])
            lssvc.sv_y = np.array(model_json['parameters']['sv_y'])
            lssvc.y_labels = np.array(model_json['parameters']['y_labels'])

        return lssvc
        
