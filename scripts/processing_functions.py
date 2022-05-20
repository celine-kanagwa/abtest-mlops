import os
import sys
import pandas as pd
import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt
import random
from datetime import timedelta
import random
import math
from data_summarizing_functions import DataSummarizer
sumar = DataSummarizer()

class DataProcessor:
    
    def load_data(self, path):
        original_df = pd.read_csv(path+"/SmartAd_original_data.csv")
        print("data loaded successfully!")
        return original_df
    
    def show_info(self, df):
        # Taking a look at the data

        print("the data has "+str(df.shape[0])+" rows, and "+str(df.shape[1])+" columns")
        
        print("\n Dataset information \n")
        print(df.info())
    
        return df

    def add_datetime(self, df):
        def turn_hour(x):
            if(x < 10):
                return "0"+str(x)+":00:00"
            else:
                return str(x)+":00:00" 

        df["hour"] = df.apply(lambda x: turn_hour(x["hour"]), axis=1)
        df["datetime"] = (df["date"]+ "-"+ df["hour"]).astype(str)
        df["datetime"].map(lambda x: pd.Timestamp(x, tz=None).strftime('%Y-%m-%d:%H'))

        return df

    def generate_bern_series(self, engagment_list, success_list):
        ber_ser = []

        for e, s in zip(engagment_list, success_list) :
            no_list = [0] * (e-s)
            yes_list = [1] * (s)
            series_item = no_list + yes_list
            random.shuffle(series_item)
            ber_ser += series_item
        return ber_ser

    def transform_data(self, df):
        df = df.copy()

        df = self.add_datetime(df)

        control_df = df.loc[df["experiment"] == "control"]
        exposed_df = df.loc[df["experiment"] == "exposed"]
        print("dataframe splitted")

        cont_date_aggr = sumar.find_agg(control_df, ["datetime"], ["yes", "no"], ["sum", "count"], ["success", "engagement"])
        expo_date_aggr = sumar.find_agg(control_df, ["datetime"], ["yes", "no"], ["sum", "count"], ["success", "engagement"])

        cont_bern = self.generate_bern_series(cont_date_aggr["engagement"].to_list(), cont_date_aggr["success"].to_list())
        expo_bern = self.generate_bern_series(expo_date_aggr["engagement"].to_list(), expo_date_aggr["success"].to_list())


        return np.array(cont_bern), np.array(expo_bern)
    
    def clean_missing(self, df):

        print ("Missing values: ", df.loc[((df["yes"]== 0) & (df["no"]==0))].shape[0])

        clean_df = df.loc[~((df["yes"]== 0) & (df["no"]==0))]

        print("Usable rows: ", clean_df.shape[0])

        return clean_df
    

    


class ConditionalSPRT:

    def __init__(self):
        pass

    def get_output(self, res):
        outcome,n, k,l,u,truncated,truncate_decision,x1,r,stats,limits = res
        output = {
            "Test": "Sequential AB testing",
            "outcome": outcome,
            "numberOfObservation": len(n)       
        }
        return output
    
    def plot_output(self, res):
        outcome,n, k,l,u,truncated,truncate_decision,x1,r,stats,limits = res
        lower = limits[:, 0]
        upper = limits[:,1]

        fig, ax = plt.subplots(figsize=(12,7))

        ax.plot(n, x1, label='Cumlative value of yes+no')

        ax.plot(n, lower, label='Lower Bound')
        ax.plot(n, upper, label='Upper Bound')

        plt.legend()


        plt.show()

    def conditionalSPRT(self, compiled,t1,alpha=0.05,beta=0.10,stop=None):
        x, y = compiled
        print("control df received", len(y))
        print("exposed df received", len(x))
        print("or, alpha, beta: ", t1, alpha, beta)
        
        if t1<=1:
            print('warning',"Odd ratio should exceed 1.")
        if (alpha >0.5) | (beta >0.5):
            print('warning',"Unrealistic values of alpha or beta were passed."
                    +" You should have good reason to use large alpha & beta values")
        if stop!=None:
            stop=math.floor(n0)

        def comb(n, k):
            return math.factorial(n) // math.factorial(k) // math.factorial(n - k)

        def lchoose(b, j):
            a=[]
            if (type(j) is list) | (isinstance(j,np.ndarray)==True):
                if len(j)<2:
                    j=j[0]
            if (type(j) is list) | (isinstance(j,np.ndarray)==True):
                for k in j:
                    n=b
                    if (0 <= k) & (k<= n):
                        a.append(math.log(comb(n,k)))
                    else:
                        a.append(0)
            else:
                n=b
                k=j
                if (0 <= k) & (k<= n):
                    a.append(math.log(comb(n,k)))
                else:
                    a.append(0)

            return np.array(a)

        def g(x,r,n,t1,t0=1):
            return -math.log(h(x,r,n,t1))+math.log(h(x,r,n,t0))

        def h(x,r,n,t=1):
        
            return f(r,n,t,offset=ftermlog(x,r,n,t))

        def f(r,n,t,offset=0):
            upper=max(0,r-n)
            lower=min(n,r)
            rng=list(range(upper,lower+1))
            return np.sum(fterm(rng,r,n,t,offset))

        def fterm(j,r,n,t,offset=0):
            ftlog=ftermlog(j,r,n,t,offset)
            return np.array([math.exp(ex) for ex in ftlog])

        def ftermlog(j,r,n,t,offset=0):
        
            xx=r-j
            lch=lchoose(n,j)
            lchdiff=lchoose(n,xx)
            lg=np.array(j)*math.log(t)
            lgsum=lch+lchdiff
            lgsum2=lgsum+lg
            lgdiff=lgsum2-offset

            return lgdiff

        def logf(r,n,t,offset=0):
        
            z=f(r,n,t,offset)
            if z>0:
                return math.log(z)
            else:
                return np.nan

        def clowerUpper(r,n,t1c,t0=1,alpha=0.05,beta=0.10):
        
            offset=ftermlog(math.ceil(r/2),r,n,t1c)
            z=logf(r,n,t1c,logf(r,n,t0,offset)+offset)
            a=-math.log(alpha/(1-beta))
            b=math.log(beta/(1-alpha))
            lower=b
            upper=1+a
            return (np.array([lower,upper])+z)/math.log(t1c/t0)

        l=math.log(beta/(1-alpha))
        u=-math.log(alpha/(1-beta))
        sample_size=min(len(x),len(y))
        n=np.array(range(1,sample_size+1))

        if stop!=None:
            n=np.array([z for z in n if z<=stop])
        x1=np.cumsum(x[n-1])
        r=x1+np.cumsum(y[n-1])
        stats=np.array(list(map(g,x1, r, n, [t1]*len(x1)))) #recurcively calls g
        #
        # Perform the test by finding the first index, if any, at which `stats`
        # falls outside the open interval (l, u).
        #
        clu=list(map(clowerUpper,r,n,[t1]*len(r),[1]*len(r),[alpha]*len(r), [beta]*len(r)))
        limits=[]
        for v in clu:
            inArray=[]
            for vin in v:
                inArray.append(math.floor(vin))
            limits.append(np.array(inArray))
        limits=np.array(limits)

        k=np.where((stats>=u) | (stats<=l))
        cvalues=stats[k]
        if cvalues.shape[0]<1:
            k= np.nan
            outcome='Unable to conclude.Needs more sample.'
        else:
            k=np.min(k)
            if stats[k]>=u:
                outcome=f'Exposed group produced a statistically significant increase.'
            else:
                outcome='Their is no statistically significant difference between two test groups'
        if (stop!=None) & (k==np.nan):
        #
        # Truncate at trial stop, using Meeker's H0-conservative formula (2.2).
        # Leave k=NA to indicate the decision was made due to truncation.
        #
            c1=clowerUpper(r,stop,t1,alpha,beta)
            c1=math.floor(np.mean(c1)-0.5)
            if x1[0]<=c1:
                truncate_decision='h0'
                outcome='Maximum Limit Decision. The aproximate decision point shows their is no statistically significant difference between two test groups'
            else:
                truncate_decision='h1'
                outcome=f'Maximum Limit Decision. The aproximate decision point shows exposed group produced a statistically significant increase.'
            truncated=stop
        else:
            truncate_decision='Non'
            truncated=np.nan
        return (outcome,n, k,l,u,truncated,truncate_decision,x1,r,stats,limits)

