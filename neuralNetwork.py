from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

def neuralNetwork(X_train_smote, y_train_smote, X_val, y_val):
    
    # Hyperparameters to experiment with
    hidden_layer_sizes_options = [(512, 256), (256, 128)]
    activation_options = ['relu', 'tanh']
    alpha_options = [0.0001, 0.001]
    
    #Setting best score and model variables
    best_f1 = 0
    best_nn_model = None
    
    

    #Loops to test all the hyperparameter combinations. 
    for hidden_layer_sizes in hidden_layer_sizes_options:
        for activation in activation_options:
            for alpha in alpha_options:
                
                # Defining the neural network model using MLPClassifier
                nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, 
                                   alpha=alpha, random_state=42, )
                
                nn.fit(X_train_smote, y_train_smote)
                
                # Predicting on validation set
                y_val_pred = nn.predict(X_val)
                
                # Computing the f1 score
                f1 = f1_score(y_val, y_val_pred, average='weighted')
                print(f"Neural Network F1-Score: {f1}, with model: {nn}")
                
                #Updating variables if the F1-Score is better on this model than the current best model
                if f1 > best_f1:
                    best_f1 = f1
                    best_nn_model = nn
    
    return best_f1, best_nn_model
    


