import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  
from randomForestClassifier import randomForest
from svm import svmClassifier
from neuralNetwork import neuralNetwork
from sklearn.metrics import classification_report, confusion_matrix
import itertools


def display_misclassified_images(X_test, y_test, y_pred, num_images=8):
    """
    Displays misclassified images in a 2x4 grid.
    """
    
    misclassified_indices = np.where(y_test != y_pred)[0]
    
    # If there are fewer misclassified images than num_images, adjust num_images
    if len(misclassified_indices) < num_images:
        num_images = len(misclassified_indices)
    
    # Create a 2x4 grid
    plt.figure(figsize=(15, 8))
    
    for i, index in enumerate(misclassified_indices[:num_images]):
        plt.subplot(2, 4, i+1)  # 2 rows, 4 columns
        plt.imshow(X_test[index].reshape(20, 20), cmap='gray') 
        plt.title(f"True: {y_test[index]}, Pred: {y_pred[index]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plots the confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()




def main(): 
    # Load the data
    X = np.load("emnist_hex_images.npy")
    y = np.load("emnist_hex_labels.npy")

    # Preprocess the data
    X = X / 255.0  # Scale pixel values to [0, 1]

    # Splitting the data into train+validation and test sets first
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Further splitting the train+validation set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    # Applying SMOTE to oversample the minority classes in the training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Running the classifiers and keeping the best F1-Score and model for each one. 
    best_nn_f1_score, best_nn_model = neuralNetwork(X_train_smote, y_train_smote, X_val, y_val)
    best_rf_f1_score, best_rf_model = randomForest(X_train_smote, y_train_smote, X_val, y_val)
    best_SVM_f1_score, best_SVM_model = svmClassifier(X_train_smote, y_train_smote, X_val, y_val)
    
    print(f"Best Neural Netowork F1-Score: {best_nn_f1_score} with model: {best_nn_model}.")
    print(f"Best Random Forest F1-Score: {best_rf_f1_score} with model: {best_rf_model}.")
    print(f"Best SVM F1-Score: {best_SVM_f1_score} with model: {best_SVM_model}.")
    
   
 
    # Comparing the F1-Scores to find the best classifier. 
    if best_rf_f1_score > best_SVM_f1_score and best_rf_f1_score > best_nn_f1_score:
       best_classifier = best_rf_model
       print(f"Random Forest has the highest F1-sSore: {best_rf_f1_score} with model: {best_classifier}")

    elif best_SVM_f1_score > best_rf_f1_score and best_SVM_f1_score > best_nn_f1_score:
        best_classifier = best_SVM_model
        print(f"SVM has the highest F1-Score: {best_SVM_f1_score} with model: {best_classifier}")
        

    else:
       best_classifier = best_nn_model
       print(f"Neural Network has the highest F1-Score: {best_nn_f1_score} with model: {best_classifier}")
        

    
    # Training the best classifier on the combined training and validation sets
    best_classifier.fit(X_train_val, y_train_val)
    
    # Testing the best classifier on the test set
    y_pred = best_classifier.predict(X_test)
    
    # Evaluating the model on the test set
    print("\nTest Set Evaluation:")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    unique_labels = np.unique(y_test)
    plot_confusion_matrix(cm, classes=unique_labels, title='Confusion Matrix')

    # Look at misclassified images. 
    display_misclassified_images(X_test, y_test, y_pred, num_images=8)
    
    
if __name__ == "__main__":
    main() 
    