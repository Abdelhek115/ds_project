import pandas as pd
import numpy as np
import random
import sys


#randomly sample k developers from user map, df1
def sampleDevs(dir_,file_,k):
    df = pd.read_csv(dir_+file_)
    df.columns = ['person_id','name','email']
    df = df.loc[df['author_name'].isin(author_list)]
    indices = random.sample(df.index,k)
    return df.ix[indices]['name'].values

#df = 'githubdownloadedprojects.commit_history_new.csv'
def extractLogFiles(df,author_list):
    #join df1 and df2
    res = df.loc[df['author_name'].isin(author_list)]
    return res


def filterByCount(df,colVal,threshold):
    df1 = df.groupby(colVal)['project'].nunique()
    authors1 = df1[df1>=threshold].index
    
    df1 = df.groupby(colVal)['file_name'].nunique()
    authors2 = df1[df1>=threshold].index
    authors = authors2.intersection(authors1)
    return extractLogFiles(df,authors)


def extractDF(df,proj_threshold):
    t = df.groupby(['project'])['commit_name'].count()
    proj_index = t[((t>=proj_threshold)&(t<=proj_threshold*3))].index
    #print proj_index
    commit_names =proj_index
    return df.loc[df['project'].isin(commit_names)]

def aggregateLogs():
    k = 40
    #arg = sys.argv
    #if arg:
    #    k = int(arg)
    dir_ = '/bigtemp/dSar/dl_githubdownloadedprojects/'
    file1 = 'githubdownloadedprojects.commit_history_new.csv'
    file2 = 'githubdownloadedprojects.user_map.csv'
    #author_list = sampleDevs(dir_,file2,k)
    df_chunks  = pd.read_csv(dir_+file1,chunksize = 1000000)
    df_final = pd.DataFrame()
    proj_threshold = 300
    commit_threshold = 50
    for i,df in enumerate(df_chunks):
        print i
        #res = extractLogFiles(df,author_list)
        res = extractDF(df,proj_threshold)
        
        df_final = pd.concat([df_final,res])
        print len(df_final)
    
    
    
    df_final = filterByCount(df_final,'commit_name',commit_threshold)
    df_final = filterByCount(df_final,'commit_name',commit_threshold)
    #df_final = df_final.ix[:,xrange(4,15)]
    df_final.to_csv("commit_history_dev_threshold_"+str(proj_threshold)+"_"+str(commit_threshold)+".csv")

    print 'done'

def main():
    aggregateLogs()

if __name__ == '__main__':
    main()


