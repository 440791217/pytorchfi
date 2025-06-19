import os
import json
import nm.models

##get orginal data file name
def getFile1(name,suffix=''):
    if suffix=='':
        fp=os.path.join('out','tmp','{}.json'.format(name))
    else:
        fp=os.path.join('out','tmp','{}_{}.json'.format(name,suffix))
    return fp

def readJson(fp,length):
    with open(fp,'r') as rf:
        dataList=json.load(rf)
    dataList=dataList[0:length]
    return dataList

def exe():
    length=1000
    models=nm.models.getModels()
    for model in models:
        name=model['name']
        fp1=getFile1(name=name)
        fp2=getFile1(name=name,suffix=length)
        dataList=readJson(fp=fp1,length=length)
        with open(fp2,'w') as wf:
            json.dump(dataList,wf,indent=2)
    pass

if __name__=='__main__':
    exe()