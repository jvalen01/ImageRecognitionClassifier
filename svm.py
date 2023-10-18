from sklearn.svm import SVC
from sklearn.metrics import f1_score

def svmClassifier(X_train_smote, y_train_smote, X_val, y_val):

    # Defining a set of parameters to check
    C_options = [0.1, 1]
    kernel_options = ['linear', 'rbf']
    
    # Variables to store the best score and model for this classifier
    best_f1_SVM_score = 0
    best_SVM_model = None

    for C in C_options:
        for kernel in kernel_options:

            # Training the SVM
            svc = SVC(C=C, kernel=kernel, random_state=42, class_weight='balanced')
            svc.fit(X_train_smote, y_train_smote)

            # Predicting on the validation set
            y_val_pred = svc.predict(X_val)

            # Calculating the f1_score
            f1 = f1_score(y_val, y_val_pred, average='weighted')
            print(f"SVM F1-Score: {f1}, with model: {svc}")

            # Updating variables if score is better than the current best model.
            if f1 > best_f1_SVM_score:
                best_f1_SVM_score = f1
                best_SVM_model = svc
                
    
    return best_f1_SVM_score, best_SVM_model
