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
    
    if pair not in edges:
        adjList[v1].append(v2)
        adjList[v2].append(v1)
    edges[pair].append(item)


def findCountDevs(df):
    t=df.transpose()
    t=t.apply(np.sum,axis=1).to_frame()
    t.columns = ['count']
    return t




def socialLinks(df):
    #df = df.transpose()
    edges = collections.defaultdict(list) #count cocurrence
    adjList = collections.defaultdict(list)
    for i in (df.index):
        row = df.loc[i]
        item = i
        row = row[row>0]
        indices = row.index
        if len(indices)>=2:
            
            for pair in list(itertools.combinations(indices, 2)):
                addPair(pair,edges,adjList,item)


    #df.apply(socialLink_row,axis=0)
    return edges,adjList




def user_feature(df,title):
    t = df.groupby('commit_name')['language'].all().reset_index()
    t.columns = ['commit_name','language']
    t.pivot_table(rows='commit_name',cols='language',aggfunc=len,fill_value=0)
    t.to_csv(title+'_user_languages_user_feautre.csv')

def item_feature(colVal,df,title):
    #language
    t = df.groupby(colVal)['language'].all().reset_index()
    t.columns = [colVal,'language']
    t.pivot_table(rows=colVal,cols='language',aggfunc=len,fill_value=0)
    t.to_csv(title+'_'+colVal+'_languages_item_feature.csv')

    #count_devs
    t = df.groupby(colVal)['commit_name'].count().reset_index()
    t.columns = [colVal,'counts']
    t.to_csv(title+'_'+colVal+'_num_commit_name_item_feature.csv')
    
    #deleted lines
    t = df.groupby(colVal)['deleted_lines'].sum().reset_index()
    t.to_csv(title+'_'+colVal+'_deleted_lines_item_feature.csv')
    
    #added lines
    t = df.groupby(colVal)['added_lines'].sum().reset_index()
    t.to_csv(title+'_'+colVal+'_added_lines_item_feature.csv')
    #bugs
    t = df.groupby(colVal)['is_bug'].sum().reset_index()
    t.to_csv(title+'_'+colVal+'_num_bugs_item_feature.csv')
    #counts
    t = df.groupby(colVal)['is_bug'].count().reset_index()
    t.columns = [colVal,'counts']
    t.to_csv(title+'_'+colVal+'count_feature.csv')




def calcSocialLink(adjList,edges,df_item,dev_item_time,colVal):
    social_links = collections.defaultdict(int)
    for k,v in adjList.items():
        for i in v:
            pair = orderedPair((k,i))
            for item in edges[pair]:
                
                time_k = dev_item_time[((dev_item_time['commit_name']==k)&(dev_item_time[colVal]==item))]['commit_date'].values

                if time_k:
                    time_k = time_k[0]
                    n_commiters = df_item[df_item['commit_date']<time_k]['commit_name'].unique()
                    if len(n_commiters)>0:
                        social_links[(k,item)]+=1.0/len(n_commiters)
    return social_links

def project_filelist(df,title):
    #project, file list
    t = df.groupby('project')['file_name'].apply(lambda x:x.tolist())
    t.to_csv(title+'.csv')

def createDataFrames(threshold,names,dataframe_list):
    
    
    for n,df in zip(names,dataframe_list):
        
        print n
        
        file_df = user_itemMatrix(df,'file_name',n)
        project_df = user_itemMatrix(df,'project',n)

        #file_df=itemFreq_filter(file_df,threshold,n+'_file')
        #project_df=itemFreq_filter(project_df,threshold,n+'_project')
        

        #user-item first time
        user_item_file = df.sort(['commit_date']).groupby(['commit_name','file_name'])['commit_date'].first().to_frame().reset_index()
        user_item_file.to_csv(n+'_user_file_time.csv')
        user_item_project = df.sort(['commit_date']).groupby(['commit_name','project'])['commit_date'].first().to_frame().reset_index()
        user_item_project.to_csv(n+'_user_project_time.csv')

        #features
        user_feature(df,n+'_file_')
        user_feature(df,n+'_project_')
        item_feature('file_name',df,n)
        item_feature('project',df,n)
        
        
        
        findCountDevs(file_df).to_csv('file_name_'+n+'dev_count.csv')
        findCountDevs(project_df).to_csv('project'+n+'dev_count.csv')
        
        edges,adjList = socialLinks(file_df.transpose())
        cPickle.dump(edges,open(n+'_edges_'+'file'+'.p','w'))
        cPickle.dump(adjList,open(n+'_adjList_'+'file'+'.p','w'))
        social_links=calcSocialLink(adjList,edges,df,user_item_file,"file_name")
        cPickle.dump(social_links,open(n+'_social_links_'+'file'+'.p','w'))


        edges,adjList = socialLinks(project_df.transpose())
        cPickle.dump(edges,open(n+'_edges_'+'project'+'.p','w'))
        cPickle.dump(adjList,open(n+'_adjList_'+'project'+'.p','w'))
        social_links=calcSocialLink(adjList,edges,df,user_item_project,"project")
        cPickle.dump(social_links,open(n+'_social_links_'+'project'+'.p','w'))
        print social_links
        #sumUsers_filter(file_df,threshold,n+'_file')
        #sumUsers_filter(project_df,threshold,n+'_project')

        #item_user(file_df,n+'_file_user_matrix_'+str(threshold))
        #item_user(project_df,n+'_project_user_matrix_'+str(threshold))




def dataframe_create(filename):
    df =pd.read_csv(filename)
    df = df.ix[:,xrange(1,len(df.columns))]
    columns = df.columns


    threshold = 5
    train_df = filterbyTime(df,'1998-08-08','2013-01-01')
    test_df = filterbyTime(df,'2013-01-01','2015-09-01')
    names = ['train','test']
    createDataFrames(threshold,names,[train_df,test_df])


    

project_filelist(pd.read_csv('commit_history_dev_threshold_300_50.csv'),'project_filelist.csv')
dataframe_create('commit_history_dev_threshold_300_50.csv')
