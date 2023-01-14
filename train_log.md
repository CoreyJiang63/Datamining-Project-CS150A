## Performance
Decision Tree 0.5927489783638191

KNN error 0.39992491787826867

Dummy Regression error 0.3927966234985601

RandomForest error 0.5164275819971735

XGBoost error 0.5466249160062124

Adaboost error 0.5382797121241062

Gradient Decision Tree error 0.5507297915523577

Logistic Regression error 0.43495883620084

MLPRegressor error 0.387724036278484

tree bagging error 0.5173741160668709

knn bagging error 0.3894057380532917

## Normalized Result:
MLPRegressor error 0.38955283124050977

Logistic Regression error 0.43495883620084

RandomForest error 0.3685936261810034(Significant!)

Decision Tree error 0.5022472023339226

KNN error 0.3981941217026672

Dummy Regression error 0.3927966234985601

XGBoost error 0.427999045773683

Adaboost error 0.38108937717196817

Gradient Decision Tree error 0.4155390146215788

knn bagging error 0.3860914329463872

tree bagging error 0.36569588934452557(Significant!)

## Optimization

1、RandomForestRegressor

 Initial RandomForest error is 0.5164275819971735

 After normalization, the optimized RandomForest error is 0.3685936261810034

 We optimize RandomForestClassifier‘s parameters 

 The parameters are n_estimators、max_depth、max_leaf_nodes、min_samples_split

 For  n_estimators , we give the range (10, 310, 50) to it.(This range means starting from 10 and ending at     360 with an interval of 50 ).

After calculating and comparing , the optimized RandomForest error is 0.367235,the best parameters {'n_estimators': 260}.

Then we give the range (240, 280, 10) to it.

After calculating and comparing，the best parameters still is  {'n_estimators': 260}.

For max_depth,we give the range (3, 23, 5) to it.

After calculating and comparing , the optimized RandomForest error is 0.365143,the best parameters 

{'max_depth': 18}.

For max_leaf_nodes,we give the range(500,1300,200) to it.

After calculating and comparing ,the optimized RandomForest error is 0.359925,the best parameters 

{'max_leaf_nodes': 1100}

“n_estimators = 260
max_depth = range(8, 23, 5)
min_samples_split = range(10, 26, 4)
max_leaf_nodes = range(500,1300,200)
forest_par = {'max_leaf_nodes':max_leaf_nodes}
model = GridSearchCV(RandomForestRegressor(n_estimators=260,max_depth=18),forest_par, n_jobs=-1)
model.fit(train_norm_x, train_y)
result = model.predict(test_norm_x)
print("optimze RandomForest error is %f" % loss_function(result, test_y))
print("best parameters", model.best_params_ )”



For min_samples_split,we give the range(14,30,4) to it.

After calculating and comparing ,the optimized RandomForest error is 0.359236,the best parameters

{'min_samples_split': 14}

"

n_estimators = 260
max_depth = range(8, 23, 5)
min_samples_split = range(10, 26, 4)
max_leaf_nodes = range(500,1300,200)
min_samples_split = range(14,30,4)
forest_par = {'min_samples_split':min_samples_split}
model = GridSearchCV(RandomForestRegressor(n_estimators=260,max_depth=18,max_leaf_nodes=1100),forest_par, n_jobs=-1)
model.fit(train_norm_x, train_y)
result = model.predict(test_norm_x)
print("optimze RandomForest error is %f" % loss_function(result, test_y))
print("best parameters", model.best_params_ )

"

2、Tree Bagging

Initial Tree Bagging error is 0.5173741160668709

 After normalization, the optimized tree bagging error is 0.36569588934452557

At first, we optimize DecisionTree's parameters 

The parameters are criterion、splitter、max_depth、min_samples_leaf

We give the range (4, 16) to max_depth and range (1,9) to  min_samples_leaf

After calculating and comparing , we get the best tree error is 0.450225

And the parameters is {'criterion': 'entropy', 'max_depth': 14, 'min_samples_leaf': 7, 'splitter': 'best'}. 

But as we use the parameters above , the error is 0.378820, which is worse than before.

So we decide to use initial DecisionTree's parameters.

Then, we optimize Tree Bagging‘s parameters 

The parameters are n_estimators、max_samples、max_features

For  n_estimators,  we give the range (150, 300, 50) to it.(This range means starting from 150 and ending at     300 with an interval of 50 ).

After calculating and comparing , the optimized tree error is 0.367053,the best parameters 

{'n_estimators': 250}

For max_samples,we give the range  [0.7,0.8,0.9,1.0] to it.

After calculating and comparing , the optimized tree error is 0.367473,the best parameters 

{'max_samples': 0.7},almost the same error as above.

For max_features,we give the range [0.8,0.9,1.0] to it.

After calculating and comparing , the optimized tree error is 0.363042,the best parameters 

{'max_samples': 0.6}.

