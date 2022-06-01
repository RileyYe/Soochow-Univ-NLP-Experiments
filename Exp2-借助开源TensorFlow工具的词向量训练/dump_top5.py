import json
import queue
import sys
conf = json.load(open("./conf.json"))
word2int: dict = conf['word2int']
int2word = dict([(v, k) for (k, v) in word2int.items()])
vectors: list[list] = conf['vectors']
dist = json.load(open("./dist.json"))
dists = dist['dist']
def top_5_closest(word):
    global dists
    # print('top_5_closest')
    index = word2int[word]
    res = queue.PriorityQueue(5)
    for i in range(index):
        if res.full():
            res.get_nowait()
        else:
            res.put([ -dists[str(i)+'-'+str(index)], int2word[i]])
    for i in range(index+1, len(word2int)):
        if res.full():
            res.get_nowait()
        else:
            res.put([-dists[str(index)+'-'+str(i)], int2word[i]])
    lst = []
    while not res.empty():
        item = res.get_nowait()[1]
        lst.append([item, vectors[word2int[item]]])
    return lst[::-1]

if __name__ == '__main__':
    top5 = {}
    total = len(word2int)
    current = 0
    for word, index in word2int.items():
        top5[word] = top_5_closest(word)
        current += 1
        sys.stdout.flush()
        print("Progress: {0:.2f}% processed {1}, {2} to go." .format (current/total* 100, current, total-current))
    json.dump({'top5': top5}, open("./top5.json", mode="w"))