from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Assuming you have a global variable df for the dataset
# Replace this with actual loading of your dataset
df = pd.read_csv('LoanApprovalPrediction.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    selected_model = request.form.get('model')

    if selected_model == 'logistic':
        # Handle missing values
        df.fillna(0, inplace=True)

        # Convert categorical variables to numerical using Label Encoding
        le = LabelEncoder()
        df['Education'] = le.fit_transform(df['Education'])
        df['Married'] = le.fit_transform(df['Married'])
        df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

        X = df[['Education', 'Married', 'ApplicantIncome', 'Credit_History']]
        y = df['Loan_Status']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #Create a Logistic Regression Tree model
        model = LogisticRegression()

        #Train the model on the training set
        model.fit(X_train, y_train)

        # Make predictions on the testing set
        y_pred = model.predict(X_test)

        # Evaluate the model
        confusion_matrix_str = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        classification_repo = classification_report(y_test, y_pred)

        # Save the confusion matrix visualization as an image
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 5))  # Adjust the figure size as needed
        sns.set(font_scale=1.2)  # Increase font size for better readability

        # Customize the heatmap colors and appearance
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    linewidths=.5, linecolor='black', square=True,
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])

        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('Actual', fontsize=14)

        plt.savefig('static/confusion_matrix.png')  # Save the plot to the static folder
        

        # Map numerical values to categorical for better visualization
        df['Married'] = df['Married'].map({1: 'Married', 0: 'Non-Married'})
        df['Education'] = df['Education'].map({1: 'Graduate', 0: 'Non-Graduate'})
        df['Loan_Status'] = df['Loan_Status'].map({1: 'Approved', 0: 'Not Approved'})
        df['Credit_History'] = df['Credit_History'].map({1: 'Yes', 0: 'No'})

        # Set the style of seaborn
        sns.set(style="whitegrid")

        # Create subplots
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

        # Plot 1: Loan Approval based on Marital Status
        sns.countplot(x='Married', hue='Loan_Status', data=df, ax=axes[0, 0])
        axes[0, 0].set_title('Loan Approval based on Marital Status')
        axes[0, 0].set_xlabel('Marital Status')
        axes[0, 0].set_ylabel('Count')
        

        # Plot 2: Loan Approval based on Education
        sns.countplot(x='Education', hue='Loan_Status', data=df, ax=axes[0, 1])
        axes[0, 1].set_title('Loan Approval based on Education')
        axes[0, 1].set_xlabel('Education')
        axes[0, 1].set_ylabel('Count')

        # Plot 3: Loan Approval based on Applicant Income
        sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=df, ax=axes[1, 0])
        axes[1, 0].set_title('Loan Approval based on Applicant Income')
        axes[1, 0].set_xlabel('Loan Status')
        axes[1, 0].set_ylabel('Applicant Income')

        # Plot 4: Loan Approval based on Credit History
        sns.countplot(x='Credit_History', hue='Loan_Status', data=df, ax=axes[1, 1])
        axes[1, 1].set_title('Loan Approval based on Credit History')
        axes[1, 1].set_xlabel('Credit History')
        axes[1, 1].set_ylabel('Count')

        # Adjust layout
        plt.tight_layout()

        #plot Logisticbarchart
        plt.savefig('static/logistic_barchart.png')

        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        # Plot ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Logistic Regression')
        plt.legend(loc="lower right")

        #ROC Curve for Logistic Regression
        plt.savefig('static/roccurve_logistic.png')
        

        return render_template('logistic.html',
            train_accuracy=f"Accuracy: {accuracy}",
            perc_accuracy=f"Accuracy Percentage: {accuracy * 100:.2f}%",
            confusion_matrix=confusion_matrix_str,
            classification_report=f"Classification Report: {classification_repo}")

    elif selected_model == 'random_forest':
        # Handle missing values
        df.fillna(0, inplace=True)
        # Convert categorical variables to numerical using Label Encoding
        le = LabelEncoder()
        df['Education'] = le.fit_transform(df['Education'])
        df['Married'] = le.fit_transform(df['Married'])
        df['Loan_Status'] = le.fit_transform(df['Loan_Status'])
        df['Credit_History'] = df['Credit_History'].map({'Yes': 1, 'No': 0})

        # Split the data into training and testing sets
        X = df[['Education', 'Married', 'ApplicantIncome', 'Credit_History']]
        y = df['Loan_Status']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        

        
        # Train the Random Forest Classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Evaluate the model
        y_pred_rf = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_rf)

        # Evaluate the model
        confusion_matrix_str = confusion_matrix(y_test, y_pred_rf)
        classification_rep = classification_report(y_test, y_pred_rf)

        # Set a different color palette
        sns.set_palette("pastel")

        # Create subplots with a different style
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

        # Plot 1: Loan Approval based on Marital Status using Random Forest
        sns.countplot(x='Married', hue='Loan_Status', data=df, ax=axes[0, 0], palette="Set2")
        axes[0, 0].set_title('Loan Approval based on Marital Status (Random Forest)')
        axes[0, 0].set_xlabel('Marital Status')
        axes[0, 0].set_ylabel('Count')

        # Plot 2: Loan Approval based on Education using Random Forest
        sns.countplot(x='Education', hue='Loan_Status', data=df, ax=axes[0, 1], palette="Set3")
        axes[0, 1].set_title('Loan Approval based on Education (Random Forest)')
        axes[0, 1].set_xlabel('Education')
        axes[0, 1].set_ylabel('Count')

        # Plot 3: Loan Approval based on Applicant Income using Random Forest
        sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=df, ax=axes[1, 0], palette="Dark2")
        axes[1, 0].set_title('Loan Approval based on Applicant Income (Random Forest)')
        axes[1, 0].set_xlabel('Loan Status')
        axes[1, 0].set_ylabel('Applicant Income')

        # Plot 4: Loan Approval based on Credit History using Random Forest
        sns.countplot(x='Credit_History', hue='Loan_Status', data=df, ax=axes[1, 1], palette="Paired")
        axes[1, 1].set_title('Loan Approval based on Credit History (Random Forest)')
        axes[1, 1].set_xlabel('Credit History')
        axes[1, 1].set_ylabel('Count')

        # Adjust layout
        plt.tight_layout()

        #plot Logisticbarchart
        plt.savefig('static/randomforest_barchart.png')

        # Save the confusion matrix visualization as an image
        cm = confusion_matrix(y_test, y_pred_rf)
        plt.figure(figsize=(5, 5))  # Adjust the figure size as needed
        sns.set(font_scale=1.2)  # Increase font size for better readability

        # Customize the heatmap colors and appearance
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    linewidths=.5, linecolor='black', square=True,
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])

        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('Actual', fontsize=14)

        plt.savefig('static/randomconfusion_matrix.png')  # Save the plot to the static folder

        fpr, tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        # Plot ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Random Forest')
        plt.legend(loc="lower right")
        
        #ROC Curve for Logistic Regression
        plt.savefig('static/roccurve_random.png')

        return render_template('random.html',
            train_accuracy=f"Accuracy: {accuracy}",
            perc_accuracy=f"Accuracy Percentage: {accuracy * 100:.2f}%",
            confusion_matrix=confusion_matrix_str,
            classification_report=f"Classification Report: {classification_rep}")

    else:
        return render_template('index.html', error='Invalid model selection')

if __name__ == '__main__':
    app.run(debug=True)
