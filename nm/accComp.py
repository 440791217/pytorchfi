import json
import os
import pandas as pd

def main():
    rp=os.path.join('out','accuracy')
    results={

    }
    for t in os.listdir(rp):
        modelName=t.split('.')[0]
        fp=os.path.join(rp,t)
        with open(fp,'r') as rf:
            data=json.load(rf)
            accs=[]
            for acc in data:
                accs.append(acc[1])
        results[modelName]=accs
    # 创建DataFrame
    df = pd.DataFrame(results)

    # 将DataFrame写入Excel文件
    df.to_excel('output.xlsx', index=False)
    print("Excel文件已生成！")    
    pass


if __name__=='__main__':
    main()


# # 创建示例数据
# data = {
#     '姓名': ['张三', '李四', '王五'],
#     '年龄': [25, 30, 28],
#     # '城市': ['北京', '上海', '广州']
# }

# data['城市']=['北京', '上海', '广州']

# # 创建DataFrame
# df = pd.DataFrame(data)

# # 将DataFrame写入Excel文件
# df.to_excel('output.xlsx', index=False)

# print("Excel文件已生成！")