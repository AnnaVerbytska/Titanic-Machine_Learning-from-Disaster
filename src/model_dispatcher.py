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
    "LogisticRegression": LogisticRegression(class_weight='balanced', 
                                                    C=0.1, # Lower values of C= reduce overfitting
                                                    penalty='l2',
                                                    random_state=42), 
    "DecisionTreeClassifier": DecisionTreeClassifier(class_weight='balanced', 
                                                    criterion='gini', 
                                                    min_samples_leaf=5, 
                                                    min_samples_split=6, 
                                                    max_depth=3, 
                                                    ccp_alpha=0.01, 
                                                    random_state=42), 
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, 
                                                    max_depth=4,  
                                                    min_samples_split=2, 
                                                    min_samples_leaf=1, 
                                                    bootstrap=True, 
                                                    random_state=42), 
    "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=100,            
                                                    random_state=42),
    "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=100, 
                                                    learning_rate=0.2, # Smaller rate avoids overfitting
                                                    max_depth=3, 
                                                    min_samples_split=3, 
                                                    min_samples_leaf=3, 
                                                    random_state=42)  
}   
