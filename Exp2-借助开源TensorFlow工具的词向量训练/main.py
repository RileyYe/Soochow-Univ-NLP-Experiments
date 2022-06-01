import json

if __name__ == "__main__":
    db = json.load(open("./top5.json", mode='r'))['top5']
    while True:
        query = input("输入需要查找的内容: ").lower()
        if query in db:
            res = db[query]
            print("5个与{}最相近的单词及其向量是:".format(query))
            for i in res:
                print("{:<10}{}".format(i[0], i[1]))
        else:
            print("无法查询到此单词")
