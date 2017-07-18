#!/usr/bin/env python

import sys
import datetime
import logging
import pandas as pd
import numpy as np
from scipy.optimize import fmin_ncg, fmin_tnc
from scipy.misc import factorial
from statsmodels.base.model import GenericLikelihoodModel
import numpy as np
from scipy.stats import poisson
from numpy import random
from scipy.optimize import minimize, show_options
import singlecopy_common as common
from model import Model
from poisson import Poisson_1ParamDistribution

from statsmodels.base.model import GenericLikelihoodModel

# This takes in y (the variable we are fitting) and X, the feature matrix along with the dispersion
# parameter alph to return the log liklihood.
def _ll_p2(y, X, beta):
    mu = np.exp(np.dot(X, beta))
    ll = poisson.logpmf(y, mu)
    return ll

# Generic class for maximum lilelihood for a given distribution. Here just replace __l1_nb2 with
# whatever distribution you want, in the format as above.

class Pois(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(Pois, self).__init__(endog, exog, **kwds)
    def nloglikeobs(self, params):
        beta = params[:-1]
        ll = _ll_p2(self.endog, self.exog, beta)
        return -ll
    def fit(self, start_params=None, **kwds): #, maxiter=10000, maxfun=5000, **kwds):
        if start_params == None:
            # Reasonable starting values
            start_params = np.append(np.zeros(self.exog.shape[1]), .5)
            start_params[0] = np.log(self.endog.mean())
        return super(Pois, self).fit(start_params=start_params, disp=0) #,
                                            #maxiter=maxiter, maxfun=maxfun,
                                             # **kwds)
class Poisson_Generic(Model):

    def __init__(self, prediction_date, account_label, train_weeks=52, min_weeks=20, policy='newsvendor', bump=False):
        self.prediction_date = prediction_date
        if isinstance(self.prediction_date, basestring):
            self.prediction_date = pd.to_datetime(self.prediction_date)
        self.offset_days = common.assign_offset_days(self.prediction_date)
        self.train_weeks = train_weeks
        self.start_date = self.prediction_date - datetime.timedelta(weeks=self.train_weeks)
        self.end_date = self.prediction_date-datetime.timedelta(days=self.prediction_date.weekday())-datetime.timedelta(days=1)
        self.min_weeks = min_weeks
        self.policy_algorithm = policy

        self.num_features = 1
        self.gap = 1
        self.max_demand = 30
        self.bump = bump

        self.account_label = account_label
        self.accounts = None
        self.df_in = None
        self.df_prediction = pd.DataFrame(columns=['account_id','bipad','date','predicted_demand','last_draw','sold_out','printsite'])
        self.df_policy = pd.DataFrame(columns=['account_id','bipad','date','predicted_draw'])

        self.models = {}
        self.sold_out = {}

        msg = 'poisson_generic.py: Initialized Poisson lag 1 model for date {0}, training on {1} weeks.'.format(prediction_date, train_weeks)
        print msg
        logging.info(msg)

    def query(self, limit=None, cache=False, check_sanity=True):
        self.accounts = common.get_account_group(self.account_label)

        # This functionality is here for testing purposes
        if limit is not None:
            self.accounts = self.accounts[:limit]

        msg = 'poisson_generic.py: Querying the DB'
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

    def fit(self):

        progress=0
        msg = 'poisson_generic.py: Fitting the Poisson lag 1 model'
        print msg
        logging.info(msg)
        for aa in self.accounts:
            
            progress=progress + 1
            sys.stdout.write("\rpoisson_generic.py: On account {0}".format(progress))
            sys.stdout.flush()

            bipadcount = 0
            df_sample = self.df_in[(self.df_in['account'] == aa)
                                & (self.df_in['dates'] > self.start_date)
                                & (self.df_in['dates'] <= self.prediction_date)]
            if len(df_sample) > self.min_weeks:
                unique_bipads = sorted(df_sample['bipad'].unique())
                # means_bipad = [df_sample.groupby('bipad')['sold'].mean()[bip] for bip in unique_bipads ]
                # good_guesses = [np.log(mbipad)/((self.num_features+2)*mbipad) for mbipad in means_bipad]
                for bipad in unique_bipads:
                    df_sample_day = df_sample[(df_sample['bipad'] == bipad)]
                    if len(df_sample_day) == 0:
                        msg = 'Warning, no bipad match for account {0}, bipad {1}'.format(aa, bipad)
                        print msg
                        print ''
                        logging.warning(msg)
                        continue
                    bvec = np.array(df_sample_day['sold'])
                    dvec = np.array(df_sample_day['draws'])
                    bvec = bvec.astype(int)
                    dvec = dvec.astype(int)

                    # Fill the feature matrix with the values for previous draws in the last num_features weeks, along with one year ago.
                    X = self.fill_feature_matrix(df_sample_day, bvec, dvec, self.num_features, self.gap)
                    
                    mod = Pois(bvec, X)
                    res = mod.fit()

                    # The last parameter is the over dispersion term which we don't need to reconstruct the mean.
                    xopt = res.params[0:-1]
                    # Save the model
                    if aa not in self.models:
                        self.models[aa] = {}

                    self.models[aa][bipad] = {'xopt':xopt, 'X':X}
                    bipadcount += 1
        print ''

    def predict(self):
        # TBD: make the predict date flexible (an input, by # weeks ahead?)

        msg = 'poisson_generic.py: Making the predictions'
        print msg
        logging.info(msg)
        n_bad_demand = 0
        for aa in self.accounts:
            if aa not in self.models:
                msg = 'poisson_generic.py: Warning, account {0} not found in the fitted models'.format(aa)
                print msg
                logging.warning(msg)
                continue
            df = self.df_in[np.int64(self.df_in['account']) == np.int64(aa)]
            bipad_list = df['bipad'].unique()

            for bipad in bipad_list:
                if bipad not in self.models[aa]:
                    msg = 'poisson_generic.py: Warning, bipad {0} not found in the fitted model for account {1}'.format(bipad, aa)
                    print msg
                    logging.warning(msg)
                    continue

                model = self.models[aa][bipad]

                # the features are the last data trained on.  ok?
                feats = np.squeeze(model['X'][-1:])
                pred_dem = np.exp(np.dot(model['xopt'],feats))

                # set infs to something better.  mean of the previous three weeks.
                if np.isinf(pred_dem) or np.isnan(pred_dem) or pred_dem>10*self.max_demand:
                    pred_dem = np.round(np.mean(model['X'][-3:][:,1]))
                    n_bad_demand += 1

                # Slice the appropriate data
                df_sample = self.df_in[(self.df_in['account'] == aa)
                                & (self.df_in['dates'] > self.start_date)
                                & (self.df_in['dates'] <= self.prediction_date)]
                df_sample_day = df_sample[(df_sample['bipad'] == bipad)]

                printsite = df_sample_day['printsite'].values[0]

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
                                                                'printsite': printsite},
                                                               ignore_index=True)

        msg = 'poisson_generic.py: {0} bad predicted demands set to the recent mean'.format(n_bad_demand)
        print msg
        logging.info(msg)

        return pd.DataFrame(self.df_prediction[['account_id','bipad','date','predicted_demand','last_draw','sold_out','printsite']])


    def policy(self):

        if self.policy_algorithm == 'policy2':
            msg = 'poisson_generic.py: Calculating the policy via poisson policy 2'
            print msg
            logging.info(msg)
            from poisson_policy2 import poisson_policy_2
            self.df_policy = poisson_policy_2(self.df_prediction)

        elif self.policy_algorithm == 'newsvendor':
            msg = 'poisson_generic.py: Calculating the policy via the news vendor algorithm'
            print msg
            logging.info(msg)
            import newsvendor
            self.df_policy = newsvendor.policy(self.df_prediction, Poisson_1ParamDistribution, self.models, self.bump)

        else:
            msg = 'poisson_generic.py: Policy algorithm {0} not implemented'.format(self.policy_algorithm)
            logging.error(msg)
            raise NotImplementedError, msg

        # Make the ints ints
        self.df_policy.loc[:,'account_id'] = np.int64(self.df_policy['account_id'])
        self.df_policy.loc[:,'bipad'] = np.int64(self.df_policy['bipad'])
        self.df_policy.loc[:,'predicted_draw'] = np.int64(self.df_policy['predicted_draw'])

        return pd.DataFrame(self.df_policy[['account_id','bipad','date','predicted_draw']])


    def policy_to_csv(self, filename):
        self.df_policy.to_csv(filename, header=True, index=False, date_format='%Y-%m-%d')


    def fitted_parameters(self):
        params = {'const':[], 'L1.sold':[], 'L2.sold':[]}
        for aa in self.accounts:
            if aa not in self.models:
                msg = 'poisson_generic.py: Warning, account {0} not found in the fitted models'.format(aa)
                print msg
                logging.warning(msg)
                continue
            df = self.df_in[np.int64(self.df_in['account'])==np.int64(aa)]
            bipad_list = df['bipad'].unique()

            for bipad in bipad_list:
                if bipad not in self.models[aa]:
                    msg = 'poisson_generic.py: Warning, bipad {0} not found in the fitted model for account {1}'.format(bipad, aa)
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
        X = np.zeros(shape=(len(bvec),self.num_features+2))
        for i in range(num_features+gap, len(bvec)):
            X[i] = np.append(bvec[i-gap+1-self.num_features:i-gap+1],[bvec[0-gap+1],1])
        return X

