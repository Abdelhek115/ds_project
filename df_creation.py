import pandas as pd
import numpy as np
from feature_selection import *
import itertools
import collections
import cPickle

#df = filterByCount(df,'commit_name',5)

'''
earliest day = '1998-08-18 06:20:07-07'
end day = ''2015-08-08 18:11:08-07'
#based on df, we find that: 22456 entries are earlier than 2007-01-01,2010-01-01,2015

'''


def filterbyTime(df,start,end):
    return df[((df['commit_date']>=start) & (df['commit_date']<end))]


def user_itemMatrix(df,col,title):
    #columns = np.unique(df[col].values)
    df = df.pivot_table(rows='commit_name', cols=col, aggfunc=len, fill_value=0)
    df['author_date'].to_csv(title+'_user_'+col+'_matrix.csv')
    print title+'_user_'+col+'_matrix.csv'
    return df['author_date']

def sumUsers_filter(df,threshold,title):
    #df = df.set_index(['commit_name'])
    df= df.apply(np.sum,axis=1).to_frame()
    df.columns = ['count']
    df = df[df['count']>threshold]
    df.to_csv(title+'user_total_'+str(threshold)+'.csv')
    print title+'user_total_'+str(threshold)+'.csv'
    return df


def itemFreq_filter(df,threshold,title):
    out = df.apply(np.sum,axis=0)
    df =  df.loc[:,out[out>threshold].index.values]
    df.to_csv(title+'_count_'+str(threshold)+'.csv')
    print title+'_count_'+str(threshold)+'.csv'
    return df


def item_user(df,title):
    df = df.ix[:,xrange(1,len(df.columns))]
    #df = df.set_index(['commit_name'])
    t = df.transpose()
    t.to_csv(title+'.csv')
    print title+'.csv'
    return t


def orderedPair(pair):
    v1,v2 = pair
    if v1[0]<v2[0]:
        return (v1,v2)
    return (v2,v1)


def addPair(pair,edges,adjList,item):
    pair = orderedPair(pair)
    v1,v2 = pair
    edges[pair].append(item)
    adjList[v1].append(v2)
    adjList[v2].append(v1)


def findCountDevs(df):
    t=df.transpose()
    t=t.apply(np.sum,axis=1).to_frame()
    t.columns = ['count']
    return t

def createDataFrames(threshold,names,dataframe_list):
    for n,df in zip(names,dataframe_list):
        
        print n
        
        file_df = user_itemMatrix(df,'file_name',n)
        project_df = user_itemMatrix(df,'project',n)

        file_df=itemFreq_filter(file_df,threshold,n+'_file')
        project_df=itemFreq_filter(project_df,threshold,n+'_project')

        df_file_ = df.loc[df['file_name'].isin(file_df.columns)]
        df_proj_ = df.loc[df['file_name'].isin(project_df.columns)]

        #user-item first time
        user_item_file = df_file_.sort(['commit_date']).groupby(['commit_name','file_name'])['commit_date'].first().to_frame().reset_index()
        user_item_file.to_csv(n+'_user_file_time.csv')
        user_item_project = df_proj_.sort(['commit_date']).groupby(['commit_name','project'])['commit_date'].first().to_frame().reset_index()
        user_item_project.to_csv(n+'_user_project_time.csv')


        findCountDevs(file_df).to_csv('file_name_'+n+'dev_count.csv')
        findCountDevs(project_df).to_csv('project'+n+'dev_count.csv')
        
        edges,adjList = socialLinks(file_df)
        cPickle.dump(edges,open(n+'_edges_'+'file'+'.p','w'))
        cPickle.dump(adjList,open(n+'_adjList_'+'file'+'.p','w'))


        edges,adjList = socialLinks(project_df)
        cPickle.dump(edges,open(n+'_edges_'+'project'+'.p','w'))
        cPickle.dump(adjList,open(n+'_adjList_'+'project'+'.p','w'))
        
        
        #sumUsers_filter(file_df,threshold,n+'_file')
        #sumUsers_filter(project_df,threshold,n+'_project')

        #item_user(file_df,n+'_file_user_matrix_'+str(threshold))
        #item_user(project_df,n+'_project_user_matrix_'+str(threshold))





def socialLinks(df):
    edges = collections.defaultdict(list) #count cocurrence
    adjList = collections.defaultdict(list)
    for i in (df.index):
        row = df.loc[i]
        item = row[0]
        row = row[row>0]
        
        indices = row.index[1:]
        if len(indices)>1:
            for pair in list(itertools.combinations(indices, 2)):
                addPair(pair,edges,adjList,item)


    #df.apply(socialLink_row,axis=0)
    return edges,adjList



def calculateSocialLink(adjList,edges,df_item,dev_item_time,colVal,title):
    social_links = collections.default(int)
    for k,v in enumerate(adjList):
        
        for i in v:
            pair = orderedPair((k,i))
            item = edges[i]
            time_k = dev_item_time[(dev_item_time['commit_name']==k)&(dev_item_time[colVal]==item)]['commit_date'].values[0]
            n_commiters = df_item[df_item['commit_date'<time_k]]['commit_name'].unique()
            social_links[(k,item)]+=1.0/n_commiters
    return social_links



df =pd.read_csv('commit_history_dev_15.csv')
df = df.ix[:,xrange(4,15)]
columns = df.columns


threshold = 5
train_df = filterbyTime(df,'1998-08-08','2013-01-01')
test_df = filterbyTime(df,'2013-01-01','2015-09-01')
names = ['train','test']
createDataFrames(threshold,names,[train_df,test_df])



#df = pd.read_csv('train_project_user_matrix_5.csv')
