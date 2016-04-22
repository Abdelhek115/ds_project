import pandas as pd
import numpy as np
import random
import sys


#randomly sample k developers from user map, df1
def sampleDevs(dir_,file_,k):
    df = pd.read_csv(dir_+file_)
    df.columns = ['person_id','name','email']
    indices = random.sample(df.index,k)
    return df.ix[indices]['name'].values

#df = 'githubdownloadedprojects.commit_history_new.csv'
def extractLogFiles(df,author_list):
    #join df1 and df2
    res = df.loc[df['author_name'].isin(author_list)]
    return res


def filterByCount(df,threshold):
    df1 = df.groupby('author_name')['project'].nunique()
    authors1 = df1[df1>=threshold].index
    
    df1 = df.groupby('author_name')['file_name'].nunique()
    authors2 = df1[df1>=threshold].index
    authors = authors2.intersection(authors1)
    return extractLogFiles(df,authors)



def aggregateLogs():
    k = 40
    #arg = sys.argv
    #if arg:
    #    k = int(arg)
    dir_ = '/bigtemp/dSar/dl_githubdownloadedprojects/'
    file1 = 'githubdownloadedprojects.commit_history_new.csv'
    file2 = 'githubdownloadedprojects.user_map.csv'
    author_list = sampleDevs(dir_,file2,k)
    df_chunks  = pd.read_csv(dir_+file1,chunksize = 1000000)
    df_final = pd.DataFrame()
    for i,df in enumerate(df_chunks):
        print i
        res = extractLogFiles(df,author_list)
        df_final = pd.concat([df_final,res])
    
    
    df_final = filterByCount(df_final,5)
    df_final.to_csv("commit_history_dev_"+str(k)+".csv")
    print 'done'

aggregateLogs()


