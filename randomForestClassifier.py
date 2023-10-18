from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def randomForest(X_train_smote, y_train_smote, X_val, y_val):


    # Defining a set of hyperparameters to check
    n_estimators_options = [100, 200]
    max_depth_options = [None, 10, 30]
    
    #Variables to store best score and model.
    best_f1_score = 0  
    best_rf_model = None

    # Loop over parameter combinations
    for n_estimators in n_estimators_options:
        for max_depth in max_depth_options:
            
            # Training the Random Forest Classifier
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1, class_weight='balanced')
            rf.fit(X_train_smote, y_train_smote)
            
            # Predicting on the validation set
            y_val_pred = rf.predict(X_val)
            
            # Calculating the f1_score
            f1 = f1_score(y_val, y_val_pred, average='weighted')
            print(f"Random Forest F1-Score: {f1}, with model: {rf}")
            
            # Updating variables if score is better than the current best model. 
            if f1 > best_f1_score:
                best_f1_score = f1
                best_rf_model = rf
    
    return best_f1_score, best_rf_model

    
