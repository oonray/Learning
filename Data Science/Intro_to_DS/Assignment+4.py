
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# In[22]:

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


# # Assignment 4 - Hypothesis Testing
# This assignment requires more individual learning than previous assignments - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.
# 
# Definitions:
# * A _quarter_ is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
# * A _recession_ is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
# * A _recession bottom_ is the quarter within a recession which had the lowest GDP.
# * A _university town_ is a city which has a high percentage of university students compared to the total population of the city.
# 
# **Hypothesis**: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession bottom. (`price_ratio=quarter_before_recession/recession_bottom`)
# 
# The following data files are available for this assignment:
# * From the [Zillow research data site](http://www.zillow.com/research/data/) there is housing data for the United States. In particular the datafile for [all homes at a city level](http://files.zillowstatic.com/research/public/City/City_Zhvi_AllHomes.csv), ```City_Zhvi_AllHomes.csv```, has median home sale prices at a fine grained level.
# * From the Wikipedia page on college towns is a list of [university towns in the United States](https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States) which has been copy and pasted into the file ```university_towns.txt```.
# * From Bureau of Economic Analysis, US Department of Commerce, the [GDP over time](http://www.bea.gov/national/index.htm#gdp) of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file ```gdplev.xls```. For this assignment, only look at GDP data from the first quarter of 2000 onward.
# 
# Each function in this assignment below is worth 10%, with the exception of ```run_ttest()```, which is worth 50%.

# In[23]:

# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}


# In[25]:

def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan","Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State","RegionName"]  )'''
    f={}
    s=[]
    
    with open("university_towns.txt","r") as a:
        b = pd.read_table(a,names=["Places"])
        
    l =  b[b["Places"].str.contains("edit")]
    
    for i in range(0,len(l)-1):
        if i <= len(l):
            f[l.iloc[i][0]] = b[(l.iloc[i].name)+1:(l.iloc[i+1].name)]["Places"].values
    for a,b in f.items():
        for i in b:
            s.append([a,i])
            
    df = pd.DataFrame(s,columns=["State","RegionName"])
    df["State"]= df["State"].str.replace(r" \(.*\)","").str.replace("\[.*\]","")
    df["RegionName"]= df["RegionName"].str.replace(r" \(.*\)","").str.replace("\[.*\]","")
    return df


# In[24]:

a = pd.read_excel("gdplev.xls",skiprows=219).drop(["Unnamed: 3","Unnamed: 7"],1)
a.columns = ["Year","GDP in billions of current dollars","GDP in billions of chained 2009 dollars","Year.1","GDP in billions of current dollars.1","GDP in billions of chained 2009 dollars.1"]   
a["Pct_change.1"] = a["GDP in billions of current dollars.1"].pct_change()
a["Year.num.1"] = a["Year.1"].dropna().apply(lambda x: x[0:-2])
g = a[a["Pct_change.1"]<0].groupby("Year.num.1").size()
 
def get_recession_start():
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''
    

    return a[(a["Year.num.1"]>= g.index.min()) & (a["Year.num.1"]<=g.index.max())].iloc[2]["Year.1"]


# In[26]:

def get_recession_end():
    '''Returns the year and quarter of the recession end time as a 
    string value in a format such as 2005q3'''
 
    return a[(a["Year.num.1"]>=g.index.min()) & (a["Year.num.1"]<=g.index.max())].iloc[-1]["Year.1"]  


# In[27]:

def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a 
    string value in a format such as 2005q3'''
    return a[a["GDP in billions of current dollars.1"] == a[a["Pct_change.1"]<0]["GDP in billions of current dollars.1"].min()]["Year.1"].values[0] 


# In[32]:

years =[ i for i in  range(1997,2016)]
twothousands = [ i for i in  range(2000,2015)]
months =[ i for i in   range(10,13)]
months += [ "0{}".format(i) for i in   range(1,10)]
remove = ["{}-{}".format(i,f) for f in months for i in years]
drop = ["{}-{}".format(i,f) for f in months for i in years ]
quarters = [str(i)+"q{}".format(f) for f in range(1,5) for i in twothousands]
quarters += ["2016q1","2016q2","2016q3"]

def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.
    
    '''

    
    a = pd.read_csv("City_Zhvi_AllHomes.csv")
    a["State"] = a["State"].replace(states)
    a = a.set_index(["State","RegionName"])
    
    for i in quarters:
        yr = i.split("q")[0]
        qt = i.split("q")[1]
        try:
            if int(qt) ==1:
                    a[i] = a[[yr+"-01",yr+"-02",yr+"-03"]].mean(axis=1)
            elif int(qt) ==2:                
                   a[i] = a[[yr+"-04",yr+"-05",yr+"-06"]].mean(axis=1) 
            elif int(qt) ==3:
                    a[i] = a[[yr+"-07",yr+"-08",yr+"-09"]].mean(axis=1)
            elif int(qt) ==4:
                    a[i] =a[[yr+"-10",yr+"-11",yr+"-12"]].mean(axis=1)
            else: pass
        except:
             a[i] = a[[yr+"-07",yr+"-08"]].mean(axis=1)
        
        
    a = a.drop(remove+["1996-04","1996-05","1996-06","1996-07","1996-08","1996-09","1996-10","1996-11","1996-12",
                       "2016-01","2016-02","2016-03","2016-04","2016-05","2016-06","2016-07","2016-08",],1)
    
    return a


# In[ ]:




# In[ ]:




# In[75]:

def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    
    
    u= get_list_of_university_towns()

    resession = (get_recession_start(),
                get_recession_end(),
                get_recession_bottom())
    
    qtr = quarters[quarters.index(resession[0]):quarters.index(resession[1])]
    
    hdq = convert_housing_data_to_quarters().reset_index()

    ut = pd.merge(u,hdq, how='inner', left_on=["State","RegionName"],right_on=["State","RegionName"])
    ut["Ratio"] = ut['2008q2'].divide(ut[resession[2]])
  
    for i in a.index:
        state = ut.loc[i,"State"]
        uta = ut.loc[i,"RegionName"]
        
        df = hdq[hdq["State"]==state]
        df = df[df["RegionName"]==uta]
        try:
            hdq = hdq.drop(df.index.unique()[0])
        except:continue
 
        nonut = pd.merge(hdq, u, how='left', on=["State", "RegionName"], indicator='Exist')

        nonut = nonut[nonut['Exist'] == 'left_only']

        nonut['Ratio'] = nonut['2008q2'].divide(nonut[resession[2]])

        statistic,pv = ttest_ind(ut['Ratio'].dropna(), nonut['Ratio'].dropna(),equal_var=True)
        
        if pv < 0.01: dif=True
        else:dif=False
            
        if statistic <0:better = "university town"
        else : better = "non-university town"
    
    return (dif, pv, better)

run_ttest()


# In[ ]:




# In[ ]:



