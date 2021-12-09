from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pickle
import os
import time

class machine_learning():
    def __init__(self,feature_metrics ,feature_selector,is_patient_wise_split = False, is_save = False, model_save_path = None):
        self.feature_metrics = feature_metrics
        self.feature_selector = feature_selector
        imputer = SimpleImputer(missing_values= np.nan , strategy= 'mean')
        self.model_save_path = model_save_path  
        self.is_save = is_save  
        if self.feature_selector.lower() == 'freq':
            self.x =feature_metrics[['rqi_fft_hrv','rqi_ar_hrv','rqi_ac_hrv','rqi_fft_rpeak','rqi_ar_rpeak','rqi_ac_rpeak',
                                         'rqi_fft_adr','rqi_ar_adr','rqi_ac_adr','activity_id']]
            self.y = feature_metrics[['error_hrv' ,'error_rpeak' ,'error_adr','patient_id']]
            self.x = np.array(self.x)
            self.y = np.array(self.y)

        elif self.feature_selector.lower() == 'morph':
            self.x = feature_metrics[['hrv_cov','hrv_mean_ptop','hrv_true_max_true_min','hrv_cov_min',
                                        'rpeak_cov','rpeak_mean_ptop','rpeak_true_max_true_min','rpeak_cov_min',
                                          'adr_cov','adr_mean_ptop','adr_true_max_true_min','adr_covn_min','activity_id']]
            self.y = feature_metrics[['error_hrv' ,'error_rpeak' ,'error_adr','patient_id']]
            self.x = np.array(self.x)
            self.y = np.array(self.y)
        elif self.feature_selector.lower() == 'freq_morph':
            self.x = feature_metrics[['rqi_fft_hrv','rqi_ar_hrv','rqi_ac_hrv','rqi_fft_rpeak','rqi_ar_rpeak','rqi_ac_rpeak',
                                      'rqi_fft_adr','rqi_ar_adr','rqi_ac_adr' ,
                                       'hrv_cov','hrv_mean_ptop','hrv_true_max_true_min','hrv_cov_min',
                                         'rpeak_cov','rpeak_mean_ptop','rpeak_true_max_true_min','rpeak_cov_min',
                                          'adr_cov','adr_mean_ptop','adr_true_max_true_min','adr_covn_min','activity_id']]
            self.y = feature_metrics[['error_hrv' ,'error_rpeak' ,'error_adr','patient_id']]
            self.x = np.array(self.x)
            self.y = np.array(self.y)
        
        imputer.fit(self.x)
        self.x = imputer.transform(self.x)
        if not(is_patient_wise_split): 
            self.x_train, self.x_test , self.y_train , self.y_test = train_test_split(self.x,self.y , test_size = 0.20, random_state = 0)
        else:
            self.x_train, self.x_test,self.y_train, self.y_test = self.patient_wise_split()
             
    def patient_wise_split(self):
        self.feature_metrics = np.array(self.feature_metrics)
        imputer_1 = SimpleImputer(missing_values= np.nan , strategy= 'mean')
        split_1 = (self.feature_metrics[:,-1])<13
        split_2 = (self.feature_metrics[:,-1])>= 13 
        if self.feature_selector.lower() == 'freq':
            x_train = self.feature_metrics[split_1, 0:9]
            x_train = np.hstack((x_train ,self.feature_metrics[split_1 , 21].reshape(-1,1)))
            y_train = self.feature_metrics[split_1,22:26]  
            x_test  = self.feature_metrics[split_2, 0:9]
            x_test = np.hstack((x_test,self.feature_metrics[split_2 , 21].reshape(-1,1)))
            y_test = self.feature_metrics[split_2, 22:26] 
        elif self.feature_selector.lower() == 'morph':
            x_train = self.feature_metrics[split_1, 9:21]
            x_train = np.hstack((x_train ,self.feature_metrics[split_1 , 21].reshape(-1,1)))
            y_train = self.feature_metrics[split_1,22:26]   
            x_test  = self.feature_metrics[split_2, 9:21]
            x_test = np.hstack((x_test,self.feature_metrics[split_2 , 21].reshape(-1,1)))
            y_test = self.feature_metrics[split_2, 22:26] 
        elif self.feature_selector.lower() == 'freq_morph':
            x_train = self.feature_metrics[split_1, 0:21]
            x_train = np.hstack((x_train ,self.feature_metrics[split_1 , 21].reshape(-1,1)))
            y_train = self.feature_metrics[split_1,22:26]  
            x_test  = self.feature_metrics[split_2, 0:21]
            x_test = np.hstack((x_test,self.feature_metrics[split_2 , 21].reshape(-1,1)))
            y_test = self.feature_metrics[split_2, 22:26]
        imputer_1.fit(x_train)
        imputer_1.fit(x_test)
        x_train = imputer_1.transform(x_train)
        x_test = imputer_1.transform(x_test)
        return x_train,x_test,y_train, y_test

    def train_test_data(self):
        '''
        Input -- None
        Output -- x_train , x_test , y_train, y_test for all the modalities corresponding to 'freq','morph',
                'freq+morph' features.
        Description -- This function divides the data into training and test set for all the modalities
        depending on the input of self.feature_selector().
        '''
        if self.feature_selector.lower() == 'freq':
            x_train_hrv = self.x_train[: , 0:3]
            x_train_rpeak = self.x_train[: ,3:6]
            x_train_adr = self.x_train[: , 6:9]
            # Divide independent variable test sets according to modulations.
            x_test_hrv = self.x_test[: , 0:3]
            x_test_rpeak = self.x_test[: , 3:6]
            x_test_adr = self.x_test[: , 6:9]
            # Divide dependent variable training sets according to modulations.
            y_train_hrv = self.y_train[: , 0].reshape(len(self.y_train) , -1)
            y_train_rpeak = self.y_train[: , 1].reshape(len(self.y_train) , -1)
            y_train_adr = self.y_train[: , 2].reshape(len(self.y_train) , -1)
            # Divide dependent variable test sets according to modulations.
            y_test_hrv = self.y_test[:,0].reshape(len(self.y_test) , -1)
            y_test_rpeak = self.y_test[: , 1].reshape(len(self.y_test) , -1)
            y_test_adr = self.y_test[: , 2].reshape(len(self.y_test) , -1)
        elif self.feature_selector.lower() == 'morph':
            x_train_hrv = self.x_train[: , 0:4]
            x_train_rpeak = self.x_train[: ,4:8]
            x_train_adr = self.x_train[: , 8:12]
            # Divide independent variable test sets according to modulations.
            x_test_hrv = self.x_test[: , 0:4]
            x_test_rpeak = self.x_test[: , 4:8]
            x_test_adr = self.x_test[: , 8:12]
            # Divide dependent variable training sets according to modulations.
            y_train_hrv = self.y_train[: , 0].reshape(len(self.y_train) , -1)
            y_train_rpeak = self.y_train[: , 1].reshape(len(self.y_train) , -1)
            y_train_adr = self.y_train[: , 2].reshape(len(self.y_train) , -1)
            # Divide dependent variable test sets according to modulations.
            y_test_hrv = self.y_test[:,0].reshape(len(self.y_test) , -1)
            y_test_rpeak = self.y_test[: , 1].reshape(len(self.y_test) , -1)
            y_test_adr = self.y_test[: , 2].reshape(len(self.y_test) , -1)
        elif self.feature_selector.lower() == 'freq_morph':
            x_train_hrv = self.x_train[: , 0:3]
            x_train_hrv = np.hstack((x_train_hrv , self.x_train[:,9:13]))
            x_train_rpeak = self.x_train[: ,3:6]
            x_train_rpeak = np.hstack((x_train_rpeak , self.x_train[:,13:17]))
            x_train_adr = self.x_train[: , 6:9]
            x_train_adr = np.hstack((x_train_adr , self.x_train[:,17:21]))
            # Divide independent variable test sets according to modulations.
            x_test_hrv = self.x_test[: , 0:3]
            x_test_hrv = np.hstack((x_test_hrv , self.x_test[:,9:13]))
            x_test_rpeak = self.x_test[: , 3:6]
            x_test_rpeak = np.hstack((x_test_rpeak , self.x_test[:,13:17]))
            x_test_adr = self.x_test[: , 6:9]
            x_test_adr = np.hstack((x_test_adr , self.x_test[:,17:21]))
            # Divide dependent variable training sets according to modulations.
            y_train_hrv = self.y_train[: , 0].reshape(len(self.y_train) , -1)
            y_train_rpeak = self.y_train[: , 1].reshape(len(self.y_train) , -1)
            y_train_adr = self.y_train[: , 2].reshape(len(self.y_train) , -1)
            # Divide dependent variable test sets according to modulations.
            y_test_hrv = self.y_test[:,0].reshape(len(self.y_test) , -1)
            y_test_rpeak = self.y_test[: , 1].reshape(len(self.y_test) , -1)
            y_test_adr = self.y_test[: , 2].reshape(len(self.y_test) , -1)
        
        return x_train_hrv,x_train_rpeak ,x_train_adr, x_test_hrv,x_test_rpeak,x_test_adr \
        ,y_train_hrv,y_train_rpeak,y_train_adr,y_test_hrv,y_test_rpeak,y_test_adr

 
    def ridge_regression (self):
        '''
        Inputs -- None
        Outputs -- y_predict_hrv -- predicted values of error corresponds to hrv. 
                   y_predict_rpeak -- predicted value of error corresponds to rpeak.
                   y_predict_adr -- predicted value of error corresponds to adr. 
                   error_hrv -- mse score of model fitting to hrv based data.
                   error_rpeak -- mse score of model fitting to rpeak based data.
                   error_adr -- mse score of model fitting to adr based data.
        Description -- This function takes the dependent and independent
                      variable and split the data into training and test set according to different modalities
                       and then applies the Ridge regression and gives the predicted values and MSE scores.
                       Also this functions tune the regularisation parameter lambda by 5 fold cross validation.
        '''
        mean_mse_hrv = []
        mean_mse_rpeak = []
        mean_mse_adr = []
        # Get the train and test data for all the modalities.
        x_train_hrv,x_train_rpeak ,x_train_adr, x_test_hrv,x_test_rpeak,x_test_adr,\
        y_train_hrv,y_train_rpeak,y_train_adr,y_test_hrv,y_test_rpeak,y_test_adr = self.train_test_data()
        # A range of lamda values which will be tuned.
        lamda = np.arange(0.01,20 , 0.01)
        # Create the lasso regression objects for each value of alpha.
        # and then run the 5 fold cross_val_predict to predict the value on training set and then calulcate the MSE.
        for i in range(len(lamda)):    
            regressor_hrv = Ridge(alpha = lamda[i])
            regressor_rpeak = Ridge(alpha = lamda[i])
            regressor_adr = Ridge(alpha = lamda[i])
            predictions_hrv = cross_val_predict(regressor_hrv , x_train_hrv , y_train_hrv , cv = 5)
            predictions_rpeak = cross_val_predict(regressor_rpeak , x_train_rpeak , y_train_rpeak , cv = 5)
            predictions_adr = cross_val_predict(regressor_adr , x_train_adr , y_train_adr , cv = 5)
            mean_mse_hrv.append(mean_squared_error(predictions_hrv, y_train_hrv)) 
            mean_mse_rpeak.append(mean_squared_error(predictions_rpeak, y_train_rpeak))
            mean_mse_adr.append(mean_squared_error(predictions_adr, y_train_adr))
        # pick out the index of minimum MSE.
        index_hrv = np.argmin(mean_mse_hrv)
        index_rpeak = np.argmin(mean_mse_rpeak)
        index_adr = np.argmin(mean_mse_adr)
        #Choose the lamda value corresponds to min MSE.
        lamda_hrv = lamda[index_hrv]
        lamda_rpeak = lamda[index_rpeak]
        lamda_adr = lamda[index_adr]
        
        regressor_hrv_new = Ridge(alpha = lamda_hrv)
        regressor_rpeak_new = Ridge(alpha = lamda_rpeak)
        regressor_adr_new = Ridge(alpha = lamda_adr)
        
        regressor_hrv_new.fit(x_train_hrv , y_train_hrv)
        regressor_rpeak_new.fit(x_train_rpeak , y_train_rpeak)
        regressor_adr_new.fit(x_train_adr , y_train_adr)

        self.save_path(regressor_hrv_new, model_type = 'Ridge_reg', modality = 'hrv')   
        self.save_path(regressor_rpeak_new, model_type = 'Ridge_reg', modality = 'rpeak_amp')   
        self.save_path(regressor_adr_new, model_type = 'Ridge_reg', modality = 'adr')   

        y_predict_hrv_new = regressor_hrv_new.predict(x_test_hrv)
        y_predict_rpeak_new = regressor_rpeak_new.predict(x_test_rpeak)
        y_predict_adr_new = regressor_adr_new.predict(x_test_adr)

        #error_hrv_new = mean_squared_error(y_test_hrv,y_predict_hrv_new)
        #error_rpeak_new = mean_squared_error(y_test_rpeak , y_predict_rpeak_new)
        #error_adr_new = mean_squared_error(y_test_adr , y_predict_adr_new)

        return y_predict_hrv_new,y_predict_rpeak_new,y_predict_adr_new #,error_hrv_new,error_rpeak_new,error_adr_new

    def cross_val_randomforest(self):
        '''
        Input -- None
        Output -- Best parameters for all the modalities and negetive mean square error.
        Description -- This functions runs 5 fold randon grid search to tune the hyperparameters
        of random forest regressor, which is to be used in random forest regression.
        '''
        # Define the nominal values of hyperparameters to be used in random forest regression. 
        n_estimators = [int(x) for x in np.arange(100 , 2000 , 50)]
        max_features = ['auto' , 'sqrt' , 'log2']
        max_depth = [int(x) for x in np.arange(10,200 ,10)]
        max_depth.append(None)
        min_samples_split = [2,4,6,10]
        min_samples_leaf = [1,2,4,6,10]
        bootstrap = [True , False]
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        # Form a grid of parameters.
        random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
        # Get the train test values of all the modalities.
        x_train_hrv,x_train_rpeak ,x_train_adr, x_test_hrv,x_test_rpeak,x_test_adr,\
        y_train_hrv,y_train_rpeak,y_train_adr,y_test_hrv,y_test_rpeak,y_test_adr = self.train_test_data()
        # Set the object for random forest regressor.
        regressor_hrv = RandomForestRegressor(random_state = 0)
        regressor_rpeak = RandomForestRegressor(random_state = 0)
        regressor_adr = RandomForestRegressor(random_state = 0)
        # Create the randomized search Cross validation object for each modality.
        rf_random_hrv = RandomizedSearchCV(estimator = regressor_hrv , param_distributions =random_grid ,
                                             n_iter= 10 , cv = 5,random_state = 0 , scoring = scorer)
        rf_random_rpeak = RandomizedSearchCV(estimator = regressor_rpeak , param_distributions =random_grid ,
                                             n_iter= 10 , cv = 5,random_state = 0, scoring = scorer)
        rf_random_adr = RandomizedSearchCV(estimator = regressor_adr , param_distributions =random_grid ,
                                             n_iter= 10 , cv = 5,random_state = 0, scoring = scorer)
        # Fit the randomized search CV objects.
        rf_random_hrv.fit(x_train_hrv , y_train_hrv.ravel())
        rf_random_rpeak.fit(x_train_rpeak , y_train_rpeak.ravel())
        rf_random_adr.fit(x_train_adr , y_train_adr.ravel())
        # Obtain the best parameters.
        best_para_hrv = rf_random_hrv.best_params_
        best_para_rpeak = rf_random_rpeak.best_params_
        best_para_adr = rf_random_adr.best_params_
        # Obtain the mean square error that has been used to tune the hyperparameters.
        score_hrv = rf_random_hrv.score(x_train_hrv , y_train_hrv)
        score_rpeak = rf_random_rpeak.score(x_train_rpeak , y_train_rpeak)
        score_adr = rf_random_adr.score(x_train_adr , y_train_adr)

        return best_para_hrv,best_para_rpeak,best_para_adr,score_hrv,score_rpeak,score_adr

    def randomforest(self):
        '''
        Inputs -- None
        Outputs -- y_predict_hrv -- predicted values of error corresponds to hrv. 
                   y_predict_rpeak -- predicted value of error corresponds to rpeak.
                   y_predict_adr -- predicted value of error corresponds to adr. 
                   error_hrv -- mse score of model fitting to hrv based data.
                   error_rpeak -- mse score of model fitting to rpeak based data.
                   error_adr -- mse score of model fitting to adr based data.
        Description -- This function takes the dependent and independent
                      variable and split the data into training and test set according to different modalities
                       and then applies the random forest regression and gives the predicted values and MSE scores.
        '''
        x_train_hrv,x_train_rpeak ,x_train_adr, x_test_hrv,x_test_rpeak,x_test_adr,\
        y_train_hrv,y_train_rpeak,y_train_adr,y_test_hrv,y_test_rpeak,y_test_adr = self.train_test_data()
        #import pdb;pdb.set_trace()
        # Create the random forest objects on the basis of parameters ontained from the cross validation functions.
        if self.feature_selector.lower() == 'freq':
            rf_reg_hrv_new = RandomForestRegressor(n_estimators = 1100 , min_samples_split = 2, min_samples_leaf = 10,
                                                bootstrap = False , max_depth = 100 , max_features = 'log2', random_state = 0)
            rf_reg_rpeak_new = RandomForestRegressor(n_estimators = 150 , min_samples_split = 10, min_samples_leaf = 10,
                                                bootstrap = True , max_depth = 100 , max_features = 'auto', random_state = 0)
            rf_reg_adr_new = RandomForestRegressor(n_estimators = 150 , min_samples_split = 10, min_samples_leaf = 10,
                                                bootstrap = True , max_depth = 100 , max_features = 'auto', random_state = 0)
        elif self.feature_selector.lower() == 'morph':
            rf_reg_hrv_new = RandomForestRegressor(n_estimators = 150 , min_samples_split = 10, min_samples_leaf = 10,
                                                bootstrap = True , max_depth = 100 , max_features = 'auto', random_state = 0)
            rf_reg_rpeak_new = RandomForestRegressor(n_estimators = 150 , min_samples_split = 10, min_samples_leaf = 10,
                                                bootstrap = True , max_depth = 100 , max_features = 'auto', random_state = 0)
            rf_reg_adr_new = RandomForestRegressor(n_estimators = 850 , min_samples_split = 4, min_samples_leaf = 1,
                                                bootstrap = True , max_depth = 190 , max_features = 'log2', random_state = 0)
        elif self.feature_selector.lower() == 'freq_morph':
            rf_reg_hrv_new = RandomForestRegressor(n_estimators = 1100 , min_samples_split = 2, min_samples_leaf = 10,
                                                bootstrap = False , max_depth = 100 , max_features = 'log2', random_state = 0)
            rf_reg_rpeak_new = RandomForestRegressor(n_estimators = 150 , min_samples_split = 10, min_samples_leaf = 10,
                                                bootstrap = True , max_depth = 100 , max_features = 'auto', random_state = 0)
            rf_reg_adr_new = RandomForestRegressor(n_estimators = 1050 , min_samples_split = 6, min_samples_leaf = 2,
                                                bootstrap = True , max_depth = None , max_features = 'auto', random_state = 0)
        # fit the model on training set
        rf_reg_hrv_new.fit(x_train_hrv , y_train_hrv.ravel())
        rf_reg_rpeak_new.fit(x_train_rpeak , y_train_rpeak.ravel())
        rf_reg_adr_new.fit(x_train_adr , y_train_adr.ravel())
        
        self.save_path(rf_reg_hrv_new, model_type = 'RF', modality = 'hrv')   
        self.save_path(rf_reg_rpeak_new, model_type = 'RF', modality = 'rpeak_amp')   
        self.save_path(rf_reg_adr_new, model_type = 'RF', modality = 'adr')   
        # predict the value on test set.
        
        y_predict_hrv = rf_reg_hrv_new.predict(x_test_hrv)
        y_predict_rpeak = rf_reg_rpeak_new.predict(x_test_rpeak)
        y_predict_adr = rf_reg_adr_new.predict(x_test_adr)
        

        decision_path_hrv = rf_reg_hrv_new.decision_path(x_test_hrv)
        decision_path_rpeak = rf_reg_rpeak_new.decision_path(x_test_rpeak)
        decision_path_adr = rf_reg_adr_new.decision_path(x_test_adr)

        print('elapesed time for RFR is {}'.format(end-start))
        print('decision_path_hrv {}'.format(decision_path_hrv[0].shape))
        print('decision_path_rpeak {}'.format(decision_path_rpeak[0].shape))
        print('decision_path_adr {}'.format(decision_path_adr[0].shape))     
        
        print('decision_path_hrv {}'.format(decision_path_hrv[1].shape))
        print('decision_path_rpeak {}'.format(decision_path_rpeak[1].shape))
        print('decision_path_adr {}'.format(decision_path_adr[1].shape))        
        # Get the MSE.
        #error_hrv = mean_squared_error(y_test_hrv , y_predict_hrv)
        #error_rpeak = mean_squared_error(y_test_rpeak , y_predict_rpeak)
        #error_adr = mean_squared_error(y_test_adr , y_predict_adr)

        return y_predict_hrv,y_predict_rpeak,y_predict_adr #,error_hrv,error_rpeak ,error_adr  

    def cross_val_sup_vec(self):
        '''
        Input -- None
        Output -- Best parameters for all the modalities and negetive of mean square error.
        Description -- This function runs 5 fold Randomgrid search to tune the hyperparameters
        of support vector regression, those hyperparameters will be used in support vector regression
        function.
        '''
        #Set the a_value and b_value which will be used to calculate C and gamma.
        a_values = np.arange(-5 , 8)
        b_values = np.arange(-13 , 6)
        # create the scorer which will be used by randomgridCV to tune the hyperparameters.
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        # Calculate the nominal hyperparameters.
        C = list(map(lambda x:2**int(x) , a_values))
        gamma = list(map(lambda x:2**int(x) , b_values))
        random_grid = {'C': C , 'gamma':gamma}
        # Get the x_train,x_test,y_train,y_test for all the modalities.
        x_train_hrv,x_train_rpeak ,x_train_adr, x_test_hrv,x_test_rpeak,x_test_adr,\
        y_train_hrv,y_train_rpeak,y_train_adr,y_test_hrv,y_test_rpeak,y_test_adr = self.train_test_data()
        # Create the SVR objects.
        svr_reg_hrv = SVR(kernel = 'rbf')
        svr_reg_rpeak = SVR(kernel = 'rbf')
        svr_reg_adr = SVR(kernel = 'rbf')
        # Create the Randomsearch CV objects.
        svr_random_hrv = RandomizedSearchCV(estimator = svr_reg_hrv , param_distributions =random_grid ,
                                            n_iter = 10 , cv = 5 , random_state = 0 , scoring = scorer)
        svr_random_rpeak = RandomizedSearchCV(estimator = svr_reg_rpeak , param_distributions =random_grid ,
                                            n_iter = 10,cv = 5, random_state = 0, scoring = scorer)
        svr_random_adr = RandomizedSearchCV(estimator = svr_reg_adr , param_distributions =random_grid ,
                                            n_iter = 10, cv = 5, random_state = 0, scoring = scorer)
        #fit the randomsearch CV objects.
        svr_random_hrv.fit(x_train_hrv , y_train_hrv.ravel())      
        svr_random_rpeak.fit(x_train_rpeak, y_train_rpeak.ravel())   
        svr_random_adr.fit(x_train_adr , y_train_adr.ravel())
        #obtain the best parameters.
        best_para_hrv = svr_random_hrv.best_params_
        best_para_rpeak = svr_random_rpeak.best_params_
        best_para_adr = svr_random_adr.best_params_
        # Calculate the MSE
        score_hrv = svr_random_hrv.score(x_train_hrv , y_train_hrv)
        score_rpeak = svr_random_rpeak.score(x_train_rpeak , y_train_rpeak)
        score_adr = svr_random_adr.score(x_train_adr , y_train_adr)
        # Obtain the index of the best parameters from the original list.
        a_opt_hrv_index = np.where(np.array(C) == best_para_hrv['C'])
        a_opt_rpeak_index = np.where(np.array(C) == best_para_rpeak['C'])
        a_opt_adr_index = np.where(np.array(C) == best_para_adr['C'])
    
        b_opt_hrv_index = np.where(np.array(gamma) ==best_para_hrv['gamma'])
        b_opt_rpeak_index = np.where(np.array(gamma) ==best_para_rpeak['gamma'])
        b_opt_adr_index = np.where(np.array(gamma) ==best_para_adr['gamma'])
        # Obtain the optimal value for a_values and b_values.
        a_opt_hrv = a_values[a_opt_hrv_index]
        a_opt_rpeak = a_values[a_opt_rpeak_index]
        a_opt_adr = a_values[a_opt_adr_index]

        b_opt_hrv = b_values[b_opt_hrv_index]
        b_opt_rpeak = b_values[b_opt_rpeak_index]
        b_opt_adr = b_values[b_opt_adr_index]
        # Create a fine list for another fine search.
        a_optimal_value_hrv = [a_opt_hrv-0.75 ,a_opt_hrv-0.5,a_opt_hrv-0.25,a_opt_hrv,a_opt_hrv+0.25,a_opt_hrv+0.5,a_opt_hrv+0.75]
        a_optimal_value_rpeak = [a_opt_rpeak-0.75 ,a_opt_rpeak-0.5,a_opt_rpeak-0.25,a_opt_rpeak,
                                 a_opt_rpeak+0.25,a_opt_rpeak+0.5,a_opt_rpeak+0.75]
        
        a_optimal_value_adr = [a_opt_adr-0.75 ,a_opt_adr-0.5,a_opt_adr-0.25,a_opt_adr,
                                 a_opt_adr+0.25,a_opt_adr+0.5,a_opt_adr+0.75]

        b_optimal_value_hrv = [b_opt_hrv-0.75 ,b_opt_hrv-0.5,b_opt_hrv-0.25,b_opt_hrv,b_opt_hrv+0.25,b_opt_hrv+0.5,b_opt_hrv+0.75]
        b_optimal_value_rpeak = [b_opt_rpeak-0.75 ,b_opt_rpeak-0.5,b_opt_rpeak-0.25,
                                  b_opt_rpeak,b_opt_rpeak+0.25,b_opt_rpeak+0.5,b_opt_rpeak+0.75]
        
        b_optimal_value_adr = [b_opt_adr-0.75 ,b_opt_adr-0.5,b_opt_adr-0.25,
                                  b_opt_adr,b_opt_adr+0.25,b_opt_adr+0.5,b_opt_adr+0.75]
        # Calculate the parameter C and gamma for the optimal values of a and b.
        C_optimal_hrv = list(map(lambda x:2.0**x , a_optimal_value_hrv))
        C_optimal_rpeak = list(map(lambda x:2.0**x , a_optimal_value_rpeak))
        C_optimal_adr = list(map(lambda x:2.0**x , a_optimal_value_adr))

        gamma_optimal_hrv = list(map(lambda x:2.0**x , b_optimal_value_hrv))
        gamma_optimal_rpeak = list(map(lambda x:2.0**x , b_optimal_value_rpeak))
        gamma_optimal_adr = list(map(lambda x:2.0**x , b_optimal_value_adr))
        
        grid_hrv_new = {'C':C_optimal_hrv,'gamma': gamma_optimal_hrv}
        grid_rpeak_new = {'C':C_optimal_rpeak,'gamma': gamma_optimal_rpeak}
        grid_adr_new = {'C':C_optimal_adr,'gamma': gamma_optimal_adr}
        # Create the objects again.
        svr_reg_hrv_new = SVR(kernel = 'rbf')
        svr_reg_rpeak_new = SVR(kernel = 'rbf')
        svr_reg_adr_new = SVR(kernel = 'rbf')
        #create the new Randomsearch CV objects for the new optimal parameters.
        svr_random_hrv_new = RandomizedSearchCV(estimator = svr_reg_hrv_new , param_distributions =grid_hrv_new ,
                                            n_iter = 10 , cv = 5 , random_state = 0 , scoring = scorer)
        svr_random_rpeak_new = RandomizedSearchCV(estimator = svr_reg_rpeak_new , param_distributions =grid_rpeak_new ,
                                            n_iter = 10,cv = 5, random_state = 0, scoring = scorer)
        svr_random_adr_new = RandomizedSearchCV(estimator = svr_reg_adr_new , param_distributions =grid_adr_new ,
                                            n_iter = 10, cv = 5, random_state = 0, scoring = scorer)
        #fit the new objects
        svr_random_hrv_new.fit(x_train_hrv,y_train_hrv.ravel())
        svr_random_rpeak_new.fit(x_train_hrv,y_train_hrv.ravel())
        svr_random_adr_new.fit(x_train_hrv,y_train_hrv.ravel())
        # get the best parameters.
        best_para_hrv_new = svr_random_hrv_new.best_params_
        best_para_rpeak_new = svr_random_rpeak_new.best_params_
        best_para_adr_new = svr_random_adr_new.best_params_
        # Calulate the MSE.
        score_hrv_new = svr_random_hrv_new.score(x_train_hrv , y_train_hrv)
        score_rpeak_new = svr_random_rpeak_new.score(x_train_rpeak , y_train_rpeak)
        score_adr_new = svr_random_adr_new.score(x_train_adr , y_train_adr)
        
        return best_para_hrv_new , best_para_rpeak_new , best_para_adr_new,score_hrv_new,score_rpeak_new,score_adr_new
    
    def supportvector(self):
        '''
        Inputs -- None
        Outputs -- y_predict_hrv -- predicted values of error corresponds to hrv. 
                   y_predict_rpeak -- predicted value of error corresponds to rpeak.
                   y_predict_adr -- predicted value of error corresponds to adr. 
                   error_hrv -- mse score of model fitting to hrv based data.
                   error_rpeak -- mse score of model fitting to rpeak based data.
                   error_adr -- mse score of model fitting to adr based data.
        Description -- This function takes the dependent and independent
                      variable and split the data into training and test set according to different modalities
                       and then applies the support vector regression and gives the predicted values and MSE scores.
        '''
        # Get the x_train,x_test,y_train,y_test for all the modalities.
        x_train_hrv,x_train_rpeak ,x_train_adr, x_test_hrv,x_test_rpeak,x_test_adr,\
        y_train_hrv,y_train_rpeak,y_train_adr,y_test_hrv,y_test_rpeak,y_test_adr = self.train_test_data()

        # support vector regression objects based on the hyperparameters obtained from cross_val_sup_vec function.
        if self.feature_selector.lower() == 'freq':
            svr_hrv_new = SVR(kernel='rbf' , C = 0.35355339, gamma= 13.45434264)
            svr_rpeak_new = SVR(kernel='rbf', C = 2.37841423 , gamma= 6.72717132)
            svr_adr_new = SVR(kernel='rbf', C = 2.82842712 , gamma= 9.51365692)
        elif self.feature_selector.lower() == 'morph':
            svr_hrv_new = SVR(kernel='rbf' , C = 0.59460356, gamma= 4.75682846)
            svr_rpeak_new = SVR(kernel='rbf', C = 0.59460356 , gamma= 4.75682846)
            svr_adr_new = SVR(kernel='rbf', C = 2.82842712 , gamma= 9.51365692)
        elif self.feature_selector.lower() == 'freq_morph':
            svr_hrv_new = SVR(kernel='rbf' , C = 2.82842712, gamma= 3.36358566)
            svr_rpeak_new = SVR(kernel='rbf', C = 1.41421356 , gamma= 5.65685425)
            svr_adr_new = SVR(kernel='rbf', C = 2.82842712 , gamma= 9.51365692)

        # Fit the models.
        svr_hrv_new.fit(x_train_hrv , y_train_hrv.ravel())
        svr_rpeak_new.fit(x_train_rpeak , y_train_rpeak.ravel())
        svr_adr_new.fit(x_train_adr , y_train_adr.ravel())
        
        self.save_path(svr_hrv_new, model_type = 'SVR', modality = 'hrv')   
        self.save_path(svr_rpeak_new, model_type = 'SVR', modality = 'rpeak_amp')   
        self.save_path(svr_adr_new, model_type = 'SVR', modality = 'adr')   
        # predict the values.
        start = time.time()
        y_predict_hrv = svr_hrv_new.predict(x_test_hrv)
        y_predict_rpeak = svr_rpeak_new.predict(x_test_rpeak)
        y_predict_adr = svr_adr_new.predict(x_test_adr)
        end = time.time()
        #error_hrv = mean_squared_error(y_test_hrv , y_predict_hrv)
        #error_rpeak = mean_squared_error(y_test_rpeak , y_predict_rpeak)
        #error_adr = mean_squared_error(y_test_adr , y_predict_adr)

        return y_predict_hrv,y_predict_rpeak,y_predict_adr #,error_hrv,error_rpeak ,error_adr 

    def lasso_regression(self):
        '''
        Inputs -- None
        
        Outputs -- y_predict_hrv -- predicted values of error corresponds to hrv. 
                   y_predict_rpeak -- predicted value of error corresponds to rpeak.
                   y_predict_adr -- predicted value of error corresponds to adr. 
                   error_hrv -- mse score of model fitting to hrv based data.
                   error_rpeak -- mse score of model fitting to rpeak based data.
                   error_adr -- mse score of model fitting to adr based data.
        Description -- This function takes the dependent and independent
                      variable and split the data into training and test set according to different modalities
                       and then applies the lasso regression and gives the predicted values and MSE scores.
        '''
        mean_mse_hrv = []
        mean_mse_rpeak = []
        mean_mse_adr = []
        # Range of values of alphs for which the model will be tuned.
        alpha = np.arange(0.001,12,0.05)
        x_train_hrv,x_train_rpeak ,x_train_adr, x_test_hrv,x_test_rpeak,x_test_adr,\
        y_train_hrv,y_train_rpeak,y_train_adr,y_test_hrv,y_test_rpeak,y_test_adr = self.train_test_data()
        # Create the lasso regression objects for each value of alpha.
        # and then run the 5 fold cross_val_predict to predict the value on training set and then calulcate the MSE.
        for i in range(len(alpha)):
            lasso_reg_hrv = Lasso(alpha= alpha[i], positive=True , fit_intercept=False)
            lasso_reg_rpeak = Lasso(alpha= alpha[i], positive=True , fit_intercept=False)
            lasso_reg_adr = Lasso(alpha= alpha[i], positive=True , fit_intercept=False)
            predictions_hrv = cross_val_predict(lasso_reg_hrv , x_train_hrv , y_train_hrv , cv = 5)
            predictions_rpeak = cross_val_predict(lasso_reg_rpeak , x_train_rpeak , y_train_rpeak , cv = 5)
            predictions_adr = cross_val_predict(lasso_reg_adr , x_train_adr , y_train_adr , cv = 5)
            mean_mse_hrv.append(mean_squared_error(predictions_hrv, y_train_hrv)) 
            mean_mse_rpeak.append(mean_squared_error(predictions_rpeak, y_train_rpeak))
            mean_mse_adr.append(mean_squared_error(predictions_adr, y_train_adr))
        # Pick the index for minimum MSE.
        index_hrv = np.argmin(mean_mse_hrv)
        index_rpeak = np.argmin(mean_mse_rpeak)
        index_adr = np.argmin(mean_mse_adr)
        # Pick the optimal value of alpha corresponding to the min MSE.
        alpha_hrv = alpha[index_hrv]
        alpha_rpeak = alpha[index_rpeak]
        alpha_adr = alpha[index_adr]
        # Create the model objects again and use the tuned parameter.
        lasso_reg_hrv_new = Lasso(alpha= alpha_hrv, positive=True , fit_intercept=False)
        lasso_reg_rpeak_new = Lasso(alpha= alpha_rpeak, positive=True , fit_intercept=False)
        lasso_reg_adr_new = Lasso(alpha= alpha_adr, positive=True , fit_intercept=False)
        # Fit the model.
        lasso_reg_hrv_new.fit(x_train_hrv , y_train_hrv.ravel())
        lasso_reg_rpeak_new.fit(x_train_rpeak , y_train_rpeak.ravel())
        lasso_reg_adr_new.fit(x_train_adr , y_train_adr.ravel())

        y_predict_hrv_new = lasso_reg_hrv_new.predict(x_test_hrv)
        y_predict_rpeak_new = lasso_reg_rpeak_new.predict(x_test_rpeak)
        y_predict_adr_new = lasso_reg_adr_new.predict(x_test_adr)

        #error_hrv_new = mean_squared_error(y_test_hrv,y_predict_hrv_new)
        #error_rpeak_new = mean_squared_error(y_test_rpeak , y_predict_rpeak_new)
        #error_adr_new = mean_squared_error(y_test_adr , y_predict_adr_new)

        return y_predict_hrv_new,y_predict_rpeak_new,y_predict_adr_new  #,error_hrv_new,error_rpeak_new,error_adr_new
    
    def cross_val_bayesianridge(self):
        '''
        Input -- None
        Output -- Best parameters for all the modalities and negetive of mean square error.
        Description -- This function runs 5 fold Randomgrid search to tune the hyperparameters
        of bayesian ridge regression, those hyperparameters will be used in bayesian ridge regression
        function.
        '''
        # Get the x_train,x_test,y_train,y_test for all the modalities. 
        x_train_hrv,x_train_rpeak ,x_train_adr, x_test_hrv,x_test_rpeak,x_test_adr,\
        y_train_hrv,y_train_rpeak,y_train_adr,y_test_hrv,y_test_rpeak,y_test_adr = self.train_test_data()
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        # Set the bayesian ridge regression objects.
        bayesian__ridge_hrv = BayesianRidge()
        bayesian__ridge_rpeak = BayesianRidge()
        bayesian__ridge_adr = BayesianRidge()
        # Set the a values based on which a list of hyperparameters will be created.
        a_val = np.arange(-19,-9)
        alpha_1 = list(map(lambda x:2**int(x) , a_val))
        alpha_2 = list(map(lambda x:2**int(x) , a_val))
        lambda_1 = list(map(lambda x:2**int(x) , a_val))
        lambda_2 = list(map(lambda x:2**int(x) , a_val))
        # list for number of iteration.
        n_iter = [int(x)  for x in np.arange(100,1000,10)]
        fit_intercept = [True , False]
        # Createt the grid.
        grid = {'n_iter':n_iter , 'fit_intercept':fit_intercept,'alpha_1':alpha_1,'alpha_2':alpha_2 , 'lambda_1':lambda_1,'lambda_2':lambda_2 }
        # Creat the randomsearchCV objects.
        br_random_hrv = RandomizedSearchCV(estimator = bayesian__ridge_hrv , param_distributions =grid ,
                                            n_iter = 10 , cv = 5 , random_state = 0 , scoring = scorer)
        br_random_rpeak = RandomizedSearchCV(estimator = bayesian__ridge_rpeak , param_distributions =grid ,
                                            n_iter = 10,cv = 5, random_state = 0, scoring = scorer)
        br_random_adr = RandomizedSearchCV(estimator = bayesian__ridge_adr , param_distributions =grid ,
                                            n_iter = 10, cv = 5, random_state = 0, scoring = scorer)
        # Fit the random search CV objects.
        br_random_hrv.fit(x_train_hrv , y_train_hrv.ravel())      
        br_random_rpeak.fit(x_train_rpeak, y_train_rpeak.ravel())   
        br_random_adr.fit(x_train_adr , y_train_adr.ravel())
        # Obtain the best parameters
        best_para_hrv = br_random_hrv.best_params_
        best_para_rpeak = br_random_rpeak.best_params_
        best_para_adr = br_random_adr.best_params_
        # Calculate the MSE
        score_hrv = br_random_hrv.score(x_train_hrv , y_train_hrv)
        score_rpeak = br_random_rpeak.score(x_train_rpeak , y_train_rpeak)
        score_adr = br_random_adr.score(x_train_adr , y_train_adr)

        return best_para_hrv , best_para_rpeak , best_para_adr,score_hrv,score_rpeak,score_adr


    def bayesian_ridge(self):
        '''
        Inputs -- None
        Outputs -- y_predict_hrv -- predicted values of error corresponds to hrv. 
                   y_predict_rpeak -- predicted value of error corresponds to rpeak.
                   y_predict_adr -- predicted value of error corresponds to adr. 
                   error_hrv -- mse score of model fitting to hrv based data.
                   error_rpeak -- mse score of model fitting to rpeak based data.
                   error_adr -- mse score of model fitting to adr based data.
        Description -- This function takes the dependent and independent
                      variable and split the data into training and test set according to different modalities
                       and then applies the bayesian ridge regression and gives the predicted values and MSE scores.
        '''
        # Get the x_train,x_test,y_train,y_test for all the modalities.
        x_train_hrv,x_train_rpeak ,x_train_adr, x_test_hrv,x_test_rpeak,x_test_adr,\
        y_train_hrv,y_train_rpeak,y_train_adr,y_test_hrv,y_test_rpeak,y_test_adr = self.train_test_data()
        # Create the bayesian ridge objects based on the tuned hyperparameters obtained from the cross_val function.
        if self.feature_selector.lower()=='freq':
            bayesian__ridge_hrv = BayesianRidge(fit_intercept = True , n_iter = 170,alpha_1 = 0.0009765625,
                                                    alpha_2 = 7.62939453125e-06 , lambda_1 = 1.52587890625e-05,
                                                    lambda_2 = 6.103515625e-05)
            bayesian__ridge_rpeak = BayesianRidge(fit_intercept = True , n_iter = 170,alpha_1 =  0.0009765625,
                                                    alpha_2 = 7.62939453125e-06 , lambda_1 = 1.52587890625e-05,
                                                    lambda_2 = 6.103515625e-05)
            bayesian__ridge_adr = BayesianRidge(fit_intercept = True , n_iter = 170,alpha_1 =  0.0009765625,
                                                    alpha_2 = 7.62939453125e-06 , lambda_1 = 1.52587890625e-05,
                                                    lambda_2 = 6.103515625e-05)
        elif self.feature_selector.lower()=='morph':
            bayesian__ridge_hrv = BayesianRidge(fit_intercept = True , n_iter = 170,alpha_1 =  0.0009765625,
                                                    alpha_2 = 7.62939453125e-06 , lambda_1 = 1.52587890625e-05,
                                                    lambda_2 = 6.103515625e-05)
            bayesian__ridge_rpeak = BayesianRidge(fit_intercept = True , n_iter = 450,alpha_1 =  0.00048828125,
                                                    alpha_2 = 7.62939453125e-06 , lambda_1 = 0.0009765625,
                                                    lambda_2 = 1.52587890625e-05)
            bayesian__ridge_adr = BayesianRidge(fit_intercept = True , n_iter = 450,alpha_1 =  1.9073486328125e-06,
                                                    alpha_2 = 0.00048828125 , lambda_1 = 0.0009765625,
                                                    lambda_2 = 7.62939453125e-06)
        elif self.feature_selector.lower() == 'freq_morph':
            bayesian__ridge_hrv = BayesianRidge(fit_intercept = True , n_iter = 170,alpha_1 =  0.0009765625,
                                                    alpha_2 =  7.62939453125e-06 , lambda_1 = 1.52587890625e-05,
                                                    lambda_2 =  6.103515625e-05)
            bayesian__ridge_rpeak = BayesianRidge(fit_intercept = True , n_iter = 170,alpha_1 =  0.0009765625,
                                                    alpha_2 = 7.62939453125e-06 , lambda_1 = 1.52587890625e-05,
                                                    lambda_2 = 6.103515625e-05)
            bayesian__ridge_adr = BayesianRidge(fit_intercept = True , n_iter = 170,alpha_1 =  0.0009765625,
                                                    alpha_2 = 7.62939453125e-06 , lambda_1 = 1.52587890625e-05,
                                                    lambda_2 =  6.103515625e-05)
        # Fit the model.
        bayesian__ridge_hrv.fit(x_train_hrv,y_train_hrv.ravel())
        bayesian__ridge_rpeak.fit(x_train_rpeak,y_train_rpeak.ravel())
        bayesian__ridge_adr.fit(x_train_adr,y_train_adr.ravel())
        
        self.save_path(bayesian__ridge_hrv, model_type = 'bRridge', modality = 'hrv')   
        self.save_path(bayesian__ridge_rpeak, model_type = 'bRridge', modality = 'rpeak_amp')   
        self.save_path(bayesian__ridge_adr, model_type = 'bRridge', modality = 'adr')   

        # predict the values.
        y_predict_hrv = bayesian__ridge_hrv.predict(x_test_hrv)
        y_predict_rpeak = bayesian__ridge_rpeak.predict(x_test_rpeak)
        y_predict_adr = bayesian__ridge_adr.predict(x_test_adr)

        #error_hrv = mean_squared_error(y_test_hrv , y_predict_hrv)
        #error_rpeak = mean_squared_error(y_test_rpeak , y_predict_rpeak)
        #error_adr = mean_squared_error(y_test_adr , y_predict_adr) 
        
        return y_predict_hrv,y_predict_rpeak,y_predict_adr #,error_hrv,error_rpeak ,error_adr

    def save_path(self, model_weights, model_type = 'rr', modality = 'hrv'):
        model_save_path = os.path.join(self.model_save_path, model_type + '_' + modality +  '.pkl') 
        with open(model_save_path, 'wb') as f:
            pickle.dump(model_weights, f)  