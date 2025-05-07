import os
import json
import itertools
from collections import Counter


def calAcc(fp):
    with open(fp,'r') as rf:
        dataList=json.load(rf)
        total=0
        correct=0
        for data in dataList:
            total+=1
            if data['predicted']==data['label']:
                correct+=1
    accuracy = 100 * correct / total
    print('fp={},acc={}%'.format(fp,round(accuracy,4)))

def readJson(fp):
    with open(fp,'r') as rf:
        dataList=json.load(rf)
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

        
# def nModular(fpList):
#     fp0=fpList[0]
#     fp1=fpList[1]
#     dataList0=readJson(fp0)
#     dataList1=readJson(fp1)
#     dataSize=len(dataList0)
#     total=0
#     correct=0
#     for i in range(dataSize):
#         total+=1
#         label=dataList0[i]['label']
#         predicted=dataList0[i]['predicted']
#         if dataList0[i]['predicted']!=dataList1[i]['predicted']:
#             if dataList0[i]['max_confidences']<dataList1[i]['max_confidences']:
#                 predicted=dataList1[i]['predicted']
#         if predicted==label:
#             correct+=1
#     accuracy = 100 * correct / total
#     print('acc={}%'.format(round(accuracy,4)))           

#     pass

if __name__=='__main__':
    fpList=[]
    # fp1=os.path.join('out','tmp','MobileNetV3_LargeV2.json')
    fp2=os.path.join('out','tmp','MobileNetV2.json')
    # fp4=os.path.join('out','tmp','MobileNetV3_Small.json')
    fp3=os.path.join('out','tmp','MobileNetV3_Large.json')
    # fpList.append(fp1)
    fpList.append(fp2)
    fpList.append(fp3)
    # fpList.append(fp4)
    # for fp in fpList:
    #     calAcc(fp=fp)
    nModular(fpList=fpList)
    exit(0)