from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Set random seeds for reproducibility


meta_model = LogisticRegression()


models = {
    
    "LogisticRegression": LogisticRegression(penalty='l1', C=1.0, 
                                                    solver='liblinear', 
                                                    class_weight=None,
                                                    random_state=42), # Lower values of C= reduce overfitting
    "DecisionTreeClassifier": DecisionTreeClassifier(class_weight='balanced', 
                                                     criterion='gini', 
                                                     min_samples_leaf=5, 
                                                     min_samples_split=6, 
                                                     max_depth=3, 
                                                     ccp_alpha=0.01, 
                                                     random_state=42), # prone to overfitting
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, # More trees improve stability
                                                    max_depth=4,  #Prevents overfitting
                                                    min_samples_split=2, 
                                                    min_samples_leaf=1, 
                                                    bootstrap=True, 
                                                    random_state=42), 
    "SVC": SVC(C=0.5,               # Controls margin width (lower reduces overfitting)
                kernel='rbf',       # Works well with nonlinear data
                gamma='scale', 
                probability=True,   # Needed for ensemble methods
                random_state=42),
    "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=300, # More trees for stability
                                                    max_depth=6,  # Limits overfitting
                                                    min_samples_split=8, 
                                                    min_samples_leaf=4, 
                                                    random_state=42),
    "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=200, 
                                                    learning_rate=0.1, # Smaller rate avoids overfitting
                                                    max_depth=3, 
                                                    min_samples_split=10, 
                                                    min_samples_leaf=4, 
                                                    random_state=42),
    "XGBClassifier": XGBClassifier(n_estimators=200, 
                                                    learning_rate=0.05, 
                                                    max_depth=4, 
                                                    min_child_weight=3, 
                                                    gamma=0.1, 
                                                    subsample=0.8,          # Uses 80% of data per tree
                                                    colsample_bytree=0.8,   # Uses 80% of features per tree
                                                    random_state=42, 
                                                    use_label_encoder=False, 
                                                    eval_metric='logloss'), # Avoids overfitting
    "CatBoostClassifier": CatBoostClassifier(iterations=300, 
                                                    learning_rate=0.05, 
                                                    depth=5,                   # Prevents overfitting
                                                    l2_leaf_reg=3, 
                                                    random_state=42, 
                                                    verbose=0),
    "StackingClassifier": StackingClassifier(estimators=[
        ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)), 
        ('cat', CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5, verbose=0, random_state=42)), 
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42))], 
        final_estimator=meta_model, cv=5),
        
    "VotingClassifier": VotingClassifier(
        estimators=[
        ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)),
        ('cat', CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5, verbose=0, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42))
        ],
        voting='soft'  # Probability-based voting
        ), # less flexible than StackingClassifier
}   
