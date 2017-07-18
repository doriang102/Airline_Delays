#!/usr/bin/env python

import sys
import datetime
import logging
import pandas as pd
import numpy as np
from scipy.optimize import fmin_ncg, fmin_tnc
from scipy.misc import factorial
from scipy.stats import poisson as poisson2

import singlecopy_common as common
from model import Model

class Poisson_Lag1(Model):

    def __init__(self, prediction_date, account_label, train_weeks=52, min_weeks=4, policy='newsvendor', lambd=np.zeros(7), bump=False):
        self.prediction_date = prediction_date
        if isinstance(self.prediction_date, basestring):
            self.prediction_date = pd.to_datetime(self.prediction_date)
        self.offset_days = common.assign_offset_days(self.prediction_date)
        self.train_weeks = train_weeks
        self.start_date = self.prediction_date - datetime.timedelta(weeks=self.train_weeks)
        self.end_date = self.prediction_date-datetime.timedelta(days=self.prediction_date.weekday())-datetime.timedelta(days=1)
        self.min_weeks = min_weeks
        self.policy_algorithm = policy

        self.num_features = 1 # usually lag 1
        self.n_static_features = 2 # usually 2 including last-year
        self.gap = 1
        self.max_demand = 500
        self.bump = bump
        self.lambd = np.array(lambd)

        self.account_label = account_label
        self.accounts = None
        self.df_in = None
        self.df_prediction = pd.DataFrame(columns=['account_id','bipad','date','predicted_demand','last_draw','sold_out','printsite'])
        self.df_policy = pd.DataFrame(columns=['account_id','bipad','date','predicted_draw','printsite'])

        self.models = {}
        self.sold_out = {}

        msg = 'poisson.py: Initialized Poisson lag 1 model for date {0}, training on {1} weeks.'.format(prediction_date, train_weeks)
        print msg
        logging.info(msg)

    def query(self, limit=None, cache=False, check_sanity=True):
        self.accounts = common.get_account_group(self.account_label)

        # This functionality is here for testing purposes
        if limit is not None:
            self.accounts = self.accounts[:limit]

        msg = 'poisson.py: Querying the DB'
        print msg
        logging.info(msg)
        with common.oracle_connect() as con:
            # get the data for the given accounts
            if cache:
                self.df_in = common.fetch_data_wcache(con, self.accounts, start_date=self.start_date, end_date=self.end_date, check_sanity=check_sanity)
            else:
                self.df_in = common.fetch_data(con, self.accounts, start_date=self.start_date, end_date=self.end_date, check_sanity=check_sanity)

        self.df_in = common.filter_flags(self.df_in)


    def transform(self):
        # add a field to the dataframe
        self.df_in['bought'] = self.df_in['sold'] + self.df_in['store_copy']

        # for accounts that changed from one edition to another, change the 
        # history to the new edition so that we use the old demands to predict.
        account_list = [274558006, 274558600, 274588656, 274576206, 274584549, 274559335, 274576297, 274532696]
        weekdays = np.array([pd.to_datetime(d).weekday() for d in self.df_in['dates']], dtype=int)

        # get the sundays for the given accounts
        indices = self.df_in['account'].isin(account_list) & (weekdays == 6)
        bipads = self.df_in[indices]['bipad']
        
        # make sure that it ends in 7
        indices_7 = (bipads % 10 == 7)
        if np.sum(indices_7) != len(bipads):
            print 'WARNING: poisson:transform expected sunday accounts 7 edition, but only saw {0}/{1}'.format(np.sum(indices_7), len(bipads))
        indices_tot = indices & indices_7

        new_bipads = self.df_in[indices_tot]['bipad'] - 1
        self.df_in.loc[indices_tot,'bipad'] = new_bipads

        # change sales=0 instances to a non-zero number
        # 0 demand makes for infs 
        self.df_in.loc[:,'sold'] = self.df_in['sold'].apply(lambda x: max(x, 0.001))

    def fit(self):

        # progress=0
        msg = 'poisson.py: Fitting the Poisson lag 1 model'
        print msg
        logging.info(msg)
        for aa in self.accounts:
            
            # progress=progress + 1
            # sys.stdout.write("\rpoisson.py: On account {0}".format(progress))
            # sys.stdout.flush()

            bipadcount = 0
            df_sample = self.df_in[(self.df_in['account'] == aa)
                                & (self.df_in['dates'] > self.start_date)
                                & (self.df_in['dates'] <= self.prediction_date)]

            unique_bipads = sorted(df_sample['bipad'].unique())
            means_bipad = [df_sample.groupby('bipad')['sold'].mean()[bip] for bip in unique_bipads ]
            good_guesses = [np.log(mbipad)/((self.num_features+self.n_static_features)*mbipad) for mbipad in means_bipad]
            for bipad in unique_bipads:
                df_sample_day = df_sample[(df_sample['bipad'] == bipad)]

                if len(df_sample_day) < self.min_weeks:
                    msg = 'Warning, {0} < {1} weeks of data for account {2}, bipad {1}'.format(len(df_sample_day), self.min_weeks, aa, bipad)
                    print msg
                    print ''
                    logging.warning(msg)
                    continue

                bvec = df_sample_day['sold'].values.astype(int)
                dvec = df_sample_day['draws'].values.astype(int)

                # Fill the feature matrix with the values for previous draws in the last num_features weeks, along with one year ago.
                X = self.fill_feature_matrix(df_sample_day, bvec, dvec, self.num_features, self.gap)
                try:
                    theta0 = [good_guesses[bipadcount]]*(self.num_features+self.n_static_features)
                except:
                    theta0 = [0.1]*(self.num_features+self.n_static_features)

                if np.isinf(theta0).any() or np.isnan(theta0).any() or (np.abs(theta0)>1).any():
                    theta0 = [0.1]*(self.num_features+self.n_static_features)

                bounds = []
                for i in range(self.num_features+self.n_static_features-1):
                    bounds.append((-1.0, 1.0))
                bounds.append((0.,500.))
                try:
                    xopt, nfeval, rc = fmin_tnc(self.L, theta0, fprime=self.dL, bounds=bounds, accuracy=1e-3, args=(bvec,X,dvec),disp=False)
                except:
                    theta0 = [0.1]*(self.num_features+self.n_static_features)
                    xopt, nfeval, rc = fmin_tnc(self.L, theta0, fprime=self.dL, bounds=bounds, accuracy=1e-3, args=(bvec,X,dvec),disp=False)
                    if rc not in [1,2]:
                        xopt *= np.nan

                if rc not in [1,2]:
                    xopt *= np.nan

                # Save the model
                if aa not in self.models:
                    self.models[aa] = {}

                self.models[aa][bipad] = {'xopt':xopt, 'theta0':theta0, 'X':X}
                bipadcount += 1

        print ''


    def predict(self):
        # TBD: make the predict date flexible (an input, by # weeks ahead?)

        msg = 'poisson.py: Making the predictions'
        print msg
        logging.info(msg)
        n_bad_demand = 0
        n_good_demand = 0
        for aa in self.accounts:
            if aa not in self.models:
                msg = 'poisson.py: Warning, account {0} not found in the fitted models'.format(aa)
                print msg
                logging.warning(msg)
                continue
            df = self.df_in[np.int64(self.df_in['account']) == np.int64(aa)]
            bipad_list = df['bipad'].unique()

            for bipad in bipad_list:
                if bipad not in self.models[aa]:
                    msg = 'poisson.py: Warning, bipad {0} not found in the fitted model for account {1}'.format(bipad, aa)
                    print msg
                    logging.warning(msg)
                    continue

                model = self.models[aa][bipad]

                # the features are the last data trained on.  ok?
                feats = np.squeeze(model['X'][-1:])
                pred_dem = np.exp(np.dot(model['xopt'],feats))

                # set infs to something better.  mean of the previous three weeks.
                if np.isinf(pred_dem) or np.isnan(pred_dem) or pred_dem>10*self.max_demand:
                    pred_dem = np.mean(model['X'][-3:][:,0].astype(float))
                    model['xopt'] = (1./feats)*np.log(pred_dem)/len(feats)
                    n_bad_demand += 1
                else:
                    n_good_demand += 1

                # Slice the appropriate data
                df_sample = self.df_in[(self.df_in['account'] == aa)
                                & (self.df_in['dates'] > self.start_date)
                                & (self.df_in['dates'] <= self.prediction_date)]
                df_sample_day = df_sample[(df_sample['bipad'] == bipad)]

                # The current prediction date is the same day of the week as the current bipad
                # but in the week immediately following the input prediction day
                # p_date_new = df_sample_day['dates'].max() + datetime.timedelta(weeks=1)
                # Actually, do what the AR is doing;  more robust to missing recent data in the DB
                # get the date of the following monday, tuesday, ... depending on bipad
                weekday = int(str(int(bipad))[-2]) - 1 # weekdays are stored from 1-7
                p_date_new = common.next_weekday(self.prediction_date, weekday, self.offset_days)

                self.df_prediction = self.df_prediction.append({'account_id':np.int64(aa),
                                                                'bipad':np.int64(bipad),
                                                                'date':pd.to_datetime(p_date_new),
                                                                'predicted_demand': pred_dem,
                                                                'last_draw': np.array(df_sample_day['draws'])[-1],
                                                                'sold_out': (np.array(df_sample_day['draws'])[-1]-np.array(df_sample_day['sold'])[-1] == 0),
                                                                'printsite': int(np.array(df_sample_day['printsite'])[-1])},
                                                               ignore_index=True)

        msg = 'poisson.py: {0}/{1} bad predicted demands set to the recent mean'.format(n_bad_demand, n_bad_demand+n_good_demand)
        print msg
        logging.info(msg)

        self.df_prediction['account_id'] = self.df_prediction['account_id'].astype(np.int64)
        return pd.DataFrame(self.df_prediction[['account_id','bipad','date','predicted_demand','last_draw','sold_out']])


    def policy(self):

        if self.policy_algorithm == 'policy2':
            msg = 'poisson.py: Calculating the policy via poisson policy 2'
            print msg
            logging.info(msg)
            from poisson_policy2 import poisson_policy_2
            self.df_policy = poisson_policy_2(self.df_prediction)

        elif self.policy_algorithm == 'newsvendor':
            msg = 'poisson.py: Calculating the policy via the news vendor algorithm with lambda={0}'.format(self.lambd)
            print msg
            logging.info(msg)
            import newsvendor

            self.df_policy = newsvendor.policy(self.df_prediction, Poisson_1ParamDistribution, self.models, self.lambd, self.bump)

        else:
            msg = 'poisson.py: Policy algorithm {0} not implemented'.format(self.policy_algorithm)
            logging.error(msg)
            raise NotImplementedError, msg

        # Make the ints ints
        self.df_policy.loc[:,'account_id'] = np.int64(self.df_policy['account_id'])
        self.df_policy.loc[:,'bipad'] = np.int64(self.df_policy['bipad'])
        self.df_policy.loc[:,'predicted_draw'] = np.int64(self.df_policy['predicted_draw'])

        return pd.DataFrame(self.df_policy[['account_id','bipad','date','predicted_draw','printsite']])


    def policy_to_csv(self, filename):
        self.df_policy.to_csv(filename, header=True, index=False, date_format='%Y-%m-%d')


    def fitted_parameters(self):
        params = {'const':[], 'L1.sold':[], 'L2.sold':[]}
        for aa in self.accounts:
            if aa not in self.models:
                msg = 'poisson.py: Warning, account {0} not found in the fitted models'.format(aa)
                print msg
                logging.warning(msg)
                continue
            df = self.df_in[np.int64(self.df_in['account'])==np.int64(aa)]
            bipad_list = df['bipad'].unique()

            for bipad in bipad_list:
                if bipad not in self.models[aa]:
                    msg = 'poisson.py: Warning, bipad {0} not found in the fitted model for account {1}'.format(bipad, aa)
                    print msg
                    logging.warning(msg)
                    continue

                p = self.models[aa][bipad]['xopt']

                params['L2.sold'].append(p[0])
                params['L1.sold'].append(p[1])
                params['const'].append(p[2])

        return params



    def fill_feature_matrix(self, df_sample_day, bvec, dvec, num_features, gap):
        # gap = Num weeks behind the prediction date that are used for training
        # using the week immediately before means gap = 1
        X = np.zeros(shape=(len(bvec),self.num_features+self.n_static_features))
        for i in range(num_features+gap, len(bvec)):
            X[i] = np.append(bvec[i-gap+1-self.num_features:i-gap+1],[bvec[0-gap+1],1])
            # TEMPORARY TEST: get rid of the year-ago point
            # X[i] = np.append(bvec[i-gap+1-self.num_features:i-gap+1],[1])
        return X


    # First derivative when b < d.
    def dL1(self, bvec, X, dvec, theta, lvec):
        theta_der = []
        for j in range(0,X.shape[1]):
            rvec = [lvec[i]*X[i,j]*(bvec[i]-lvec[i])/lvec[i] if bvec[i] < dvec[i] else 0 for i in range(0,len(bvec))]
            theta_der.append(np.sum(rvec))
        return theta_der

    # First derivative when b = d.
    def dL2(self, bvec, X, dvec, theta, lvec):
        theta_der = []
        for j in range(0,X.shape[1]):
            rvec = [lvec[i]*X[i,j]*(self.pcond(bvec[i],lvec[i])-lvec[i])/lvec[i] if bvec[i] >= dvec[i] else 0 for i in range(0,len(bvec))]
            theta_der.append(np.sum(rvec))
        return theta_der

    # Second derivative when b < d.
    def d2L1_hess(self, bvec,X,dvec,theta,lvec):
        Y = np.zeros(shape=(X.shape[1],X.shape[1]))
        for j in range(0, X.shape[1]):
            for k in range(0, X.shape[1]):
                Y[j,k] = np.sum([(-1)*X[i,j]*X[i,k]*lvec[i] if bvec[i] < dvec[i] else 0 for i in range(0,len(bvec))])
        return Y

    # Second derivative when b >= d.
    def d2L2_hess(self, bvec,X,dvec,theta,lvec):
        Y = np.zeros(shape=(X.shape[1],X.shape[1]))
        # precompute these arrays, since don't depend on k or j
        avg_z2_cached=np.array([self.avg_z2(bvec[i],lvec[i]) for i in range(0,len(bvec))])
        avg_z_cached=np.array([self.avg_z(bvec[i],lvec[i]) for i in range(0,len(bvec))])
        avg_I2_cached=np.array([((self.avg_I(bvec[i],lvec[i]))**2) for i in range(0,len(bvec))])
        k1=lvec * avg_z_cached / avg_I2_cached # this is just a constant


        for j in range(0, X.shape[1]):
            for k in range(j, X.shape[1]):   # since Hess symmetric only need to compute half
                # this could be vectorized but not sure worth it
                # I still don't like the O^3 loop, but we're much faster now inside, so I'll leave it as a list comp
                tvec =[ \
                        X[i,j]*X[i,k]* avg_z2_cached[i] \
                        - k1[i] \
                        - X[i,j]*X[i,k]*lvec[i]  \
                           if bvec[i] >= dvec[i] \
                        else 0  \
                        for i in range(0,len(bvec)) \
                      ]
                Y[j,k] = np.sum(tvec)
                Y[k,j] = Y[j,k]     # take advantage of symmetry

        return Y

    # Combined functions with both L1 and L2 and data.
    def L(self, theta, bvec, X, dvec):
        lvec = np.exp(np.dot(theta,X.T))
        rvec = self.L1(bvec, dvec, lvec) + self.L2(bvec, dvec, lvec)
        return (-1)*rvec

    # First part of likelihood function. Make theta a vector now. It means X needs to be a feature matrix.
    def L1(self, bvec, dvec, lvec):
        rvec = np.log(self.f(bvec,lvec))
        rvec[bvec>=dvec] = 0
        return np.sum(rvec)

    # Second part of likelihood function.
    def L2(self, bvec, dvec, lvec):
        rvec = [np.log(self.F(bvec[i],lvec[i])) if bvec[i] >= dvec[i] else 0 for i in range(0,len(bvec))]
        return np.sum(rvec)

    def dL(self, theta, bvec, X, dvec):
        lvec = np.exp(np.dot(theta,X.T))
        rvec = np.array(self.dL1(bvec,X,dvec,theta,lvec))+np.array(self.dL2(bvec,X,dvec,theta,lvec))
        return (-1)*rvec

    def d2L(self, theta, bvec, X, dvec):
        lvec = np.exp(np.dot(theta,X.T))
        a = np.matrix(self.d2L1_hess(bvec,X,dvec,theta,lvec))
        b = np.matrix(self.d2L2_hess(bvec,X,dvec,theta,lvec))
        return (-1)*(a+b)

    # Conditional probability for z > b.
    def pcond(self, b, l):
        br = np.arange(b,self.max_demand)
        rvec = br*self.p(br,l)
        return np.sum(rvec)/self.avg_I(b,l)

    # Average of z^2 in some range.
    def avg_z2(self, b, l):
        br = np.arange(b,self.max_demand)
        return np.sum( (br**2)*self.p(br,l) )

    # Average of z in some range.
    def avg_z(self, b, l):
        br = np.arange(b,self.max_demand)
        return np.sum( br*self.p(br,l) )

    # Average of indictator function in some range.
    def avg_I(self, b, l):
        return np.sum(self.p(np.arange(b,self.max_demand),l))

    # Distribution function - standard Poisson. Can be easily modified to include other parameters.
    def p(self, z, l):
        return (l**z)*(1/factorial(z))*np.exp(-l)

    # only if b < d
    def f(self, b, l):
       return self.p(b,l)

    # b >= d
    def F(self, b, l):
        rvec = self.p(np.arange(b, self.max_demand),l)
        return np.sum(rvec)


class Poisson_1ParamDistribution():
    def __init__(self, params):
        self.params = params
        feats = np.squeeze(params['X'][-1:])
        self.mu = np.exp(np.dot(params['xopt'],feats))

    def pmf(self, x):
        return poisson2.pmf(x, self.mu)

    def cdf(self, x):
        return poisson2.cdf(x, self.mu)
