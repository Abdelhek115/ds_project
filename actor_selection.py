import numpy as np
import pandas as pd
#import collections
#import heapq
import multiprocessing as mp
import cPickle



#based on number of commits, select top k files/projects/authors
class topKItems(object):
    def __init__(self,df):
        self.df = df

    def filterbyCommitTime(self,start,end):
        df = self.df
        self.df = df[df['commit_date']<=start and df['commit_date']>=end]
    
    def filterbyCreated_at(self,start,end):
        df = self.df
        self.df = df[df['created_at']<=start and df['created_at']>=end]

    def groupbyProj(self):
        return self.df.groupby(['project'])

    def groupbyDev(self):
        return self.df.groupby(['author_name'])
    
    def groupbyFile(self):
        return self.df.groupby(['file_name'])
    
    def groupbyDev_File(self):
        return self.df.groupby(['author_name','file_name'])

    def selectTopKprojtect(self,k):
        return self.groupbyProj().count().order(ascending=False).head(k)

    def selectTopKdev(self,k):
        return self.groupbyDev()['author_name'].count().order(ascending=False).head(k)

    def selectTopKfile(self,k):
        return self.groupbyFile().count().order(ascending=False).head(k)


def mp_processing(df,k=500):
    return topKItems(df).selectTopKdev(k)


dir_ = '/bigtemp/dSar/dl_githubdownloadedprojects/'
file1 = 'githubdownloadedprojects.commit_history_new.csv'
#file2 = 'githubdownloadedprojects.active_users_projects.csv'

def multiprocess(dir_,file):
    df1  = pd.read_csv(dir_+file,chunksize = 1000000)
    pool = mp.Pool(2)
    df_list  = []
    k = 500
    for i,df in enumerate(df1):
        print i
        f = pool.apply_async(mp_processing,(df,),dict(k=500))
        df_list.append(f)
    print 'done'
    print df_list[0].head()
    cPickle.dump(df_list,open('top500dev.p','w'))

multiprocess(dir_,file1)


#merge the top k data frames for each chunk
class mergeChunks(object):
    def __init__(self,listDF):
        self.listDF = listDF




