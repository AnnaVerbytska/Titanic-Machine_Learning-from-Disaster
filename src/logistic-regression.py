# Import libraries
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# DATA INGESTION

# Fetch the data
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

# PREPROCESSING

# Create a fake Survived column for test data
test_df.loc[:,"Survived"] = -1

# Concatenate both training and test data
data = pd.concat([train_df,test_df]).reset_index(drop=True)

# FEATURE ENGINEERING

# Drop non interested columns
data.drop(['PassengerId','Name','Cabin','Embarked','Sex','Ticket','Age','Fare'], axis=1, inplace=True)
data.head()

# split the training and test data again
train_df = data[data.Survived != -1].reset_index(drop=True) 
test_df = data[data.Survived == -1].reset_index(drop=True)

# CROSS-VALIDATION
# We create a new column called kfold and fill it with -1
train_df['kfold'] = -1

# The next step is to randomize the rows of the data
train_df = train_df.sample(frac=1,random_state=32).reset_index(drop=True)

# Fetch the targets
y = train_df.Survived.values

# Inititate the kfold class
kf = model_selection.StratifiedKFold(n_splits=5)

# fill the new kfold column
for f, (t_, v_) in enumerate(kf.split(X=train_df,y=y)):
    train_df.loc[v_,'kfold'] = f


# Collect accuracies
lst = []

# Loop folds
for fold in range(0,5):
    # Training data is where kfold is not equal to provided fold
    df_train = train_df[train_df.kfold != fold].reset_index(drop=True)
    
    # Validation data is where kfold is equal to provided fold
    df_valid = train_df[train_df.kfold == fold].reset_index(drop=True)

    # Drop the Survived column from dataframe and convert it to a numpy array
    x_train = df_train.drop('Survived',axis=1).values
    y_train = df_train.Survived.values

    # Similarly, for validation we have
    x_valid = df_valid.drop('Survived',axis=1).values
    y_valid = df_valid.Survived.values

    # INITIALIZE THE MODEL & FINE-TUNING
    model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', class_weight='balanced',random_state=32)

    # Fit the model on training data
    model.fit(x_train,y_train)

    # Create predictions for validations samples
    preds = model.predict(x_valid)

    # Calculate & print accuracy
    accuracy = metrics.accuracy_score(y_valid,preds)
    print(f"Fold = {fold}, Accuracy = {accuracy}")

    lst.append(accuracy)

Average = sum(lst) / len(lst) 
print(f"Average accuracy = {Average}")

# Make predictions on the test data
test_predictions = model.predict(test_df.values)

# Prepare the submission file
submission = pd.read_csv('../data/submission.csv')
submission['Survived'] = test_predictions
submission.head(20)
