import numpy as np
import pandas as pd
from scipy.misc import factorial

# Realistic maximum demand for any Starbucks.
max_demand = 25

def poisson_policy_2(df_preds):

    ## set parameters
    # set u, e.g., u = (shape=1,scale=1)

    # paraemeters for the gamma distribution from which each store's "t" is drawn:
    # NB mean(theta)=u[0]*u[1]
    # NB var(theta)=u[0]*u[1]^2
    u=(40),.1
    u=.1,40.0

    # minimum draw
    d0=1

    # ignore, a policy variable
    sigma=0

    # max # of "store copies"
    c=2

    # TBD: consolidate this with the money calculation in AR

    # MONEY parameters:
    # profit, i.e., alpha=(1+\nu)*\gamma
    alpha=427 # NDM sunday revenue
    gamma=133 # NDM sunday oversell=drop+newsprint
    nu=alpha/gamma-1
    # value of a stor copy, relative to cost of printing
    eps=1 # units of cents

    # zbar will be the mean # copies demanded.
    # if draw is super large,
    # this will be the # purchaed
    zbar=u[0]*u[1]

    # loop parameters
    r_count=20
    r_max=6.
    # for averaging:
    stores=6729

    # n is number; m is money
    # main goal of this notebook is to plot m vs n
    mvec=np.zeros(r_count)
    nvec=np.zeros(r_count)

    # not really important, just diagnostic:
    pvec=np.zeros(r_count)
    dvec=np.zeros(r_count)
    svec=np.zeros(r_count)
    rvec=np.zeros(r_count)

    # initialize vector of thetas for all the stores
    tvec=np.random.gamma  (shape=u[0],scale=u[1],size=stores)

    # now, armed with each store's theta ("t")
    # loop over stores
    account_id = df_preds['account_id'].values
    bipad = df_preds['bipad'].values
    date = df_preds['date'].values
    predictions = []
    #prediction = pd.DataFrame({'account_id':aa,  'bipad':bipad, 'date':p_date_new, 'predicted_draw': predicted_draw }, index=[0])
    zvec=np.zeros(len(df_preds))
    tvec=np.zeros(len(df_preds))

    for ids in range (0,len(df_preds)):
        # each store gets its own z (demand)
        lamd = np.float(df_preds['predicted_demand'][ids])
        if lamd > 30:
            lamd=0
        zvec[ids]=np.rint(lamd) #np.random.poisson(lam=lamd)
        tvec[ids]=np.nan_to_num(np.float(lamd))
    stores = len(df_preds)

    # loop in r
    policy_preds = []
    money_earned = 0
    money_earned_old = 0
    money_earned_array = []
    for ids in range(0,stores):
        for idr in range (0,r_count):
            # trying r
            r=float(idr)/float(r_count)*r_max

            # initialize sum for m and n
            msum=0
            nsum=0
            ## not really needed:
            psum=0
            ssum=0
            dsum=0

            # loop over stores
            #for ids in range (1,stores):
            #ids = 5
            # POLICY for a given store's (known) "t" (note z not known)
            d=dcalc(d0,r,tvec[ids],sigma)
            # resulting # purchased given store's true "z" (demand)
            p=pcalc(zvec[ids],d)
            # resulting # store copies
            s=scalc(zvec[ids],d,c)
            # total # circulation
            # NB: n=p+s (i.e., number we report is store copies+ purchased copies)
            nsum=nsum+p+s
            # total money made, normalized by printing cost
            msum=msum+np.nan_to_num((1.+nu)*p+eps*s-d)

            # not really needed:
            ssum=ssum+s
            psum=psum+p
            dsum=dsum+d

            # save for plotting later
            rvec[idr]=r
            nvec[idr]=nsum
            mvec[idr]=msum

            # not really needed:
            svec[idr]=ssum
            dvec[idr]=dsum
            pvec[idr]=psum

        nvec=np.nan_to_num(nvec) #/float(stores);
        mvec=np.nan_to_num(mvec) #/float(stores);
        # not really needed:
        svec=svec/float(stores)
        pvec=pvec/float(stores)
        dvec=dvec/float(stores)
        policy_preds.append(max(nvec[np.argmax(mvec)],1))
        money_earned = money_earned + np.nanmax(mvec)
        money_earned_array.append(money_earned)

    df_preds['policy_preds'] = policy_preds
    df_prediction = pd.DataFrame({'account_id':account_id.astype(int), 'bipad':bipad.astype(int), 'date':date, 'predicted_draw':policy_preds})

    return df_prediction

# Distribution function - standard Poisson. Can be easily modified to include other parameters.
def p(z,l):
    return (l**z)*(1/factorial(z))*np.exp(-l)

def pcalc(z,d):

    # how many copies were purchased?
    # well, if the draw was big enough it's the demand (z)
    # if not, it's the draw (d)

    if z > d:
        p = d
    else:
        p = z
    return p

def dcalc(d0,r,t,sigma):
    # the POLICY: that is,
    # we need to choose the draw for a store.
    # in this model we consider that the true demand (z)
    # will be drawn from a poisson distribution with parameter t
    # (i.e., expected # of copies is t)
    #
    # this still leaves open the question: should we send exactly t?
    # slightly more? slightly less?
    # we parameterize this by sending d=[r*t], i.e., round r*t where
    # r is a scale factor
    #
    # we include here possible "d0" -- the minimum we would possibly send.

    import numpy
    rt = r*t
    # Stores that have infinite or null demand are the onsey twosies more often than not,so
    # fix the policy to deliver the minimum draw in that case.
    import math
    if math.isnan(t):
        return 1
    if t is None:
        return 1
    if t > 100:
        return 1
    if sigma == -1:
        rt = numpy.floor(rt)
    if sigma == 0:
        rt = numpy.rint(rt)
    if sigma == 1:
        rt = numpy.ceil(rt)
        #rt = ceil(rt)
    return int(max(d0,rt))

def scalc(z,d,c):

    # how many store copies can we count?
    # well if all copies are bought (z>d) it's none (0
    # if there's no demand at all, it's either "c" (the max we are allowed)
    # or it's the number we sent, whichever is smaller

    if z>d:
        s=0
    elif z>d-c:
        s=d-z
    else:
        s=min(d,c)
    return(s)

