import os
import json
import itertools
from collections import Counter
import nm.models
import itertools
import numpy as np


dataCache={

}

def calAcc(name):
    fp=getFile1(name)
    accResult=[]
    with open(fp,'r') as rf:
        dataList=json.load(rf)
        clazzSize=len(dataList[0]['confidences'])
        # print(clazzSize)
        clazzTotal=[0]*clazzSize
        # print(clazzTotal)
        # exit(0)
        clazzCorrect=[0]*clazzSize
        clazzAcc=[0]*clazzSize
        total=0
        correct=0
        for data in dataList:
            total+=1
            clazzTotal[data['label']]+=1
            if data['predicted']==data['label']:
                correct+=1
                clazzCorrect[data['label']]+=1
        accuracy = 100 * correct / total
        print('model:{},acc:{}%'.format(name,round(accuracy,4)))
        accResult.append(['total',round(accuracy,4)])
        for i in range(clazzSize):
            clazzAcc[i]=100*clazzCorrect[i]/clazzTotal[i]
            print('model:{},class:{},acc:{}%'.format(name,i,round(clazzAcc[i],4)))
            accResult.append([i,round(clazzAcc[i],4)])
        with open(os.path.join('out','accuracy','{}.json'.format(name)),'w') as wf:
            json.dump(accResult,wf,indent=2)
    return accuracy



def readJson(fp):
    global dataCache
    if fp in dataCache.keys():
        dataList=dataCache[fp]
    else:
        with open(fp,'r') as rf:
            dataList=json.load(rf)
        dataCache[fp]=dataList
    dataList=dataList
    return dataList


def nModular(fpList):
    N=2
    combinations = list(itertools.combinations(fpList, N))
    for combs in combinations:
        dataList=[]
        total=0
        correct=0
        for comb in combs:
            dataList.append(readJson(comb))
        dataSize=len(dataList[0])
        for i in range(dataSize):
            total+=1
            label=dataList[0][i]['label']
            predicteds=[]
            max_confidences=[]
            confidences=[]
            top_5_indices=[]
            for j in range(N):
                predicteds.append(dataList[j][i]['predicted'])
                max_confidences.append(dataList[j][i]['max_confidences'])
                sorted_indices = sorted(range(len(dataList[j][i]['confidences'])), key=lambda k: dataList[j][i]['confidences'][k], reverse=True)
                top_5_indices.append(sorted_indices[:5])
                confidences.append(dataList[j][i]['confidences'])
            # 获取置信度最高的前 5 个类别的索引
            # # 获取出现次数最多的元素及其次数
            zero_list = [0] * 1000
            for j in range(N):
                top_5=top_5_indices[j]
                confidence=confidences[j]
                for t in top_5:
                    zero_list[t]+=confidence[t]
            max_value = max(zero_list)
            predicted = zero_list.index(max_value)
            # counter = Counter(predicteds)
            # most_common_element, count = counter.most_common(1)[0]
            # if count>1:
            #     predicted=most_common_element
            # else:
            #     max_value = max(max_confidences)
            #     predicted=dataList[max_confidences.index(max_value)][i]['predicted']
            # if label in top_5_indices[0]:
            if predicted==label:
                correct+=1
        accuracy = 100 * correct / total
        print('combs:',combs)
        print('acc={}%'.format(round(accuracy,4)))    

        
def ensembleModels(names):
    print("###############################Ensemble Models################################################")
    ##models size
    modelSize=len(names)
    ###read data
    dataList=[]
    for name in names:
        fp=getFile1(name=name)
        data=readJson(fp=fp)
        dataList.append(data)
    ##get data size
    dataSize=len(dataList[0])
    classSize=len(dataList[0][0]['confidences'])
    #init params
    total=0
    correct=0
    for i in range(dataSize):
        total+=1
        label=dataList[0][i]['label']
        zero_list = [0] * classSize
        for j in range(modelSize):
            zero_list[dataList[j][i]['predicted']]+=1
        max_value = max(zero_list)
        predicted=zero_list.index(max_value)
        if predicted==label:
            correct+=1
    accuracy = 100 * correct / total
    accuracy = round(accuracy,4)
    # print('combs:',combs)
    print('acc={}%'.format(accuracy))
    return 

        
def ensembleModelsByAvg(names):
    print("###############################Ensemble Models by Avg################################################")
    ##models size
    modelSize=len(names)
    ###read data
    dataList=[]
    for name in names:
        fp=getFile1(name=name)
        data=readJson(fp=fp)
        dataList.append(data)
    ##get data size
    dataSize=len(dataList[0])
    classSize=len(dataList[0][0]['confidences'])
    #init params
    total=0
    correct=0
    for i in range(dataSize):
        total+=1
        label=dataList[0][i]['label']
        confidences=np.zeros(classSize)
        for j in range(modelSize):
            confidences+=np.array(dataList[j][i]['confidences'])
        predicted = np.argmax(confidences)
        if predicted==label:
            correct+=1
    accuracy = 100 * correct / total
    accuracy = round(accuracy,4)
    # print('combs:',combs)
    print('acc={}%'.format(accuracy))
    return


def ensembleModelsByMax(names):
    print("###############################Ensemble Models by Max################################################")
    ##models size
    modelSize=len(names)
    ###read data
    dataList=[]
    for name in names:
        fp=getFile1(name=name)
        data=readJson(fp=fp)
        dataList.append(data)
    ##get data size
    dataSize=len(dataList[0])
    classSize=len(dataList[0][0]['confidences'])
    #init params
    total=0
    correct=0
    for i in range(dataSize):
        total+=1
        label=dataList[0][i]['label']
        confidences=np.zeros(classSize)
        for j in range(modelSize):
            # print(np.array(dataList[j][i]['confidences']))
            confidences=np.maximum(confidences,np.array(dataList[j][i]['confidences']))
        predicted = np.argmax(confidences)
        if predicted==label:
            correct+=1
    accuracy = 100 * correct / total
    accuracy = round(accuracy,4)
    # print('combs:',combs)
    print('acc={}%'.format(accuracy))
    return 

def ensembleModelsByMedian(names):
    print("###############################Ensemble Models by Median################################################")
    ##models size
    modelSize=len(names)
    ###read data
    dataList=[]
    for name in names:
        fp=getFile1(name=name)
        data=readJson(fp=fp)
        dataList.append(data)
    ##get data size
    dataSize=len(dataList[0])
    #init params
    total=0
    correct=0
    for i in range(dataSize):
        total+=1
        label=dataList[0][i]['label']
        confidences=[]
        for j in range(modelSize):
            confidences.append(dataList[j][i]['confidences'])
            # print(np.array(dataList[j][i]['confidences']))
            # confidences=np.maximum(confidences,np.array(dataList[j][i]['confidences']))
        stacked = np.stack(confidences, axis=0)
        median_values = np.median(stacked, axis=0)
        predicted = np.argmax(median_values)
        if predicted==label:
            correct+=1
    accuracy = 100 * correct / total
    accuracy = round(accuracy,4)
    # print('combs:',combs)
    print('acc={}%'.format(accuracy))
    return 


def ensembleModelsByVoter(names):
    print("###############################Ensemble Models by Voter################################################")
    ##models size
    modelSize=len(names)
    ###read data
    dataList=[]
    for name in names:
        fp=getFile1(name=name)
        data=readJson(fp=fp)
        dataList.append(data)
    ##get data size
    dataSize=len(dataList[0])
    classSize=len(dataList[0][0]['confidences'])
    #init params
    total=0
    correct=0
    for i in range(dataSize):
        total+=1
        label=dataList[0][i]['label']
        zero_list = [0] * classSize
        ####max
        confidences=np.zeros(classSize)
        for j in range(modelSize):
            confidences+=np.array(dataList[j][i]['confidences'])
        predicted = np.argmax(confidences)
        zero_list[predicted]+=1
        ###median
        confidences=[]
        for j in range(modelSize):
            confidences.append(dataList[j][i]['confidences'])
        stacked = np.stack(confidences, axis=0)
        median_values = np.median(stacked, axis=0)
        predicted = np.argmax(median_values)
        zero_list[predicted]+=1
        ###avg
        confidences=np.zeros(classSize)
        for j in range(modelSize):
            confidences+=np.array(dataList[j][i]['confidences'])
        predicted = np.argmax(confidences)
        zero_list[predicted]+=1
        max_value = max(zero_list)
        predicted=zero_list.index(max_value)
        if predicted==label:
            correct+=1
    accuracy = 100 * correct / total
    accuracy = round(accuracy,4)
    # print('combs:',combs)
    print('acc={}%'.format(accuracy))
    return


def ensembleModelsByBias(names,models):
    print("###############################Ensemble Models by Bias################################################")
    ##models size
    modelSize=len(names)
    ###read data
    dataList=[]
    ###weight
    accList=[]
    for name in names:
        fp=getFile1(name=name)
        data=readJson(fp=fp)
        dataList.append(data)
        for model in models:
            if model['name']==name:
                accList.append(model['acc']) 
    ##get data size
    dataSize=len(dataList[0])
    classSize=len(dataList[0][0]['confidences'])
    #init params
    total=0
    correct=0
    for i in range(dataSize):
        total+=1
        label=dataList[0][i]['label']
        ####max
        confidences=np.zeros(classSize)
        for j in range(modelSize):
            confidences+=np.array(dataList[j][i]['confidences'])*accList[j]
        predicted = np.argmax(confidences)
        if predicted==label:
            correct+=1
    accuracy = 100 * correct / total
    accuracy = round(accuracy,4)
    # print('combs:',combs)
    print('acc={}%'.format(accuracy))
    return 

# def ensembleModels1(names):
#     print("###############################Ensemble Models################################################")
#     ##models size
#     modelSize=len(names)
#     ###read data
#     dataList=[]
#     for name in names:
#         fp=getFile1(name=name,suffix=1000)
#         data=readJson(fp=fp)
#         dataList.append(data)
#     ##get data size
#     dataSize=len(dataList[0])
#     classSize=len(dataList[0][0]['confidences'])
#     #init params
#     total=0
#     correct=0
#     for i in range(dataSize):
#         total+=1
#         label=dataList[0][i]['label']
#         predicteds=[]
#         max_confidences=[]
#         confidences=[]
#         zero_list = [0] * classSize
#         for j in range(modelSize):
#             zero_list[dataList[j][i]['predicted']]+=1
#             predicteds.append(dataList[j][i]['predicted'])
#             max_confidences.append(dataList[j][i]['max_confidences'])
#             confidences.append(dataList[j][i]['confidences'])
#         max_value = max(zero_list)
#         predicted=zero_list.index(max_value)
#         if predicted==label:
#             correct+=1
#     accuracy = 100 * correct / total
#     # print('combs:',combs)
#     print('acc={}%'.format(round(accuracy,4)))   

##get orginal data file name
def getFile1(name,suffix=''):
    if suffix=='':
        fp=os.path.join('out','tmp','{}.json'.format(name))
    else:
        fp=os.path.join('out','tmp','{}_{}.json'.format(name,suffix))
    return fp

##Combination
def getCombs(elements,n):
    combinations = itertools.combinations(elements, n)
    # for comb in combinations:
    #         print(comb)
    return combinations

def entry():
    mainModelName='EfficientNet_B7'
    models=nm.models.getModels()
    modelNames=[]
    for item in models:
        modelNames.append(item['name'])
    n=3
    for name in modelNames:
        print("#########################Calculate the model's accuracy######################################################")
        acc=calAcc(name)
        for model in models:
            if model['name']==name:
                model['acc']=acc
    return
    combs=getCombs(modelNames,n)
    for comb in combs:
        # if mainModelName not in comb:
        #     continue
        names=list(comb)
        print('Model Collection:{}'.format(names))
        ensembleModels(names)
        ensembleModelsByAvg(names)
        ensembleModelsByMax(names)
        ensembleModelsByMedian(names)
        ensembleModelsByVoter(names)
        ensembleModelsByBias(names,models)

if __name__=='__main__':
    entry()
    # fpList=[]
    # # fp1=os.path.join('out','tmp','MobileNetV3_LargeV2.json')
    # fp2=os.path.join('out','tmp','MobileNetV2.json')
    # # fp4=os.path.join('out','tmp','MobileNetV3_Small.json')
    # fp3=os.path.join('out','tmp','MobileNetV3_Large.json')
    # # fpList.append(fp1)
    # fpList.append(fp2)
    # fpList.append(fp3)
    # # fpList.append(fp4)
    # # for fp in fpList:
    # #     calAcc(fp=fp)
    # nModular(fpList=fpList)
    exit(0)