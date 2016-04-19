'''After preprocessing, use created features to run models in randomforest, gradientboost, adaboost, and run grid_search
Gradientboost gave the best result
'''

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


#quickrun of randomforest to get a baseline estimate
def rf_test(X,y):

    RF_model = RandomForestRegressor(100,n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
    RF_model.fit(X_train,y_train)
    y_pred = RF_model.predict(X_test)
    print mean_squared_error(y_test, y_pred), r2_score(y_test,y_pred)


#check number of trees for randomforest, based on mse and r2
def rf_estimators(X,y,num_trees):
    '''
    input: X as input and y as target, num_trees is the list of range
    output: two lists for mse and r2
    '''
    mse=[]
    r2=[]
    for n in num_trees:
        rf = RandomForestRegressor(n_estimators=n,n_jobs=-1)
        mse_score = np.mean(cross_val_score(rf, X, y, cv=5, scoring='mean_squared_error'))*(-1)
        r2_score = np.mean(cross_val_score(rf, X, y, cv=5, scoring='r2'))
        mse.append(mse_score)
        r2.append(r2_score)
    return mse, r2


#run gridsearch
def grid_search(est, grid,X,y):
    grid_cv = GridSearchCV(est, grid, n_jobs=-1, verbose=True,
                           scoring='mean_squared_error').fit(X,y)
    return grid_cv


#run cross-validation of model, based on mean_squared_error
def cross_val(model, X, y):
    r2_score = cross_val_score(model,X,y, scoring='r2',cv=5)
    mse_score = abs( cross_val_score(model,X,y,scoring='mean_squared_error',cv=5) )
    print model.__class__.__name__+'| MSE:',mse_score.mean(),'| R2:',r2_score.mean()


#use model to predict on test data and write csv file
def make_submission(model,X):
    y_pred = model.predict(X)
    output_df= pd.DataFrame(df_test['id'],columns=['id',"relevance"])
    output_df['relevance'] = y_pred
    output_df['relevance'] = output_df['relevance'].apply(lambda x: 3 if x>3 else x)
    output_df['relevance'] = output_df['relevance'].apply(lambda x: 1 if x<1 else x)
    output_df.to_csv('submission.csv',index=False)


if __name__ = '__main__':

    df_new = pd.read_pickle('df_new.pkl')
    #take only training set
    df_new_train = df_new.iloc[:74067,:]
    y = df_new_train['relevance'].astype(float)
    cols =['ratio_title', 'ratio_description', 'ratio_brand','search_term_digit', 'word_in_title', 'word_in_description', 'word_in_brand', 'len_search', 'query_in_description']
    X = df_new_train[cols].values

    #quick run of randomforest
    rf_test(X,y)

    #check number of trees for randomforest; didn't make much difference
    num_trees = range(20,300,20)
    rf_mse, rf_r2 = rf_estimators(X,y,num_trees)

    #grid_search randomforest
    rf_grid = {'max_depth': [3, None],
           'min_samples_split': [1, 3, 10],
           'min_samples_leaf': [1, 3, 10],
           'bootstrap': [True, False],
           'n_estimators': [50,100]}
    rf_grid_search = grid_search(RandomForestRegressor(), rf_grid,X,y)
    print rf_grid_search.best_params_
    #{'min_samples_split': 3, 'n_estimators': 100, 'bootstrap': True, 'max_depth': None, 'min_samples_leaf': 10}
    print rf_grid_search.best_score_
    #0.24643947016025253

    #check gradientboost and adaboost
    gdbr = GradientBoostingRegressor(learning_rate=0.1,loss='ls',n_estimators=100)
    cross_val(gdbr,X,y)
    #GradientBoostingRegressor| MSE: 0.243163983717 | R2: 0.127430331191
    abr = AdaBoostRegressor(DecisionTreeRegressor(),learning_rate=0.1,loss='linear',
                        n_estimators=100,random_state=1)
    cross_val(abr,X,y)
    #AdaBoostRegressor| MSE: 0.25696289748 | R2: 0.0785707159071

    #run grid_search on gradientboost
    gdbr_grid = {'max_depth':[4,6],'learning_rate':[0.1,0.05,0.01],\
    'min_samples_leaf':[3,10],'max_features':[1.0,0.3], \
    'n_estimators':[100]}
    gdbr_gridsearch = GridSearchCV(GradientBoostingRegressor(),gdbr_grid,scoring='mean_squared_error')
    gdbr_gridsearch.fit(X,y)
    print gdbr_gridsearch.best_params_
    print gdbr_gridsearch.best_score_

    #use best model from randomforest and gradientboost to make submissions
    df_test = df_new.iloc[74067:,:]
    X_test = df_test[cols].values
    rf_best = rf_grid_search.best_estimator_
    make_submission(rf_best,X_test)
    #score 0.49615
    gdbr_best = gdbr_gridsearch.best_estimator_
    make_submission(gdbr_best,X_test)
    #score 0.49501
