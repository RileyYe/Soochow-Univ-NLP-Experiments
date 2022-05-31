from openpyxl import load_workbook
from jieba import cut
import re
question = []
answer = []
def participle():
    global question
    global answer
    wb = load_workbook('./问答型数据范例.xlsx')
    ws = wb['工作表1']
    question = [list(cut(str(i.value), use_paddle=True)) for i in tuple(ws.columns)[2][1::]]
    answer = [list(cut(str(i.value), use_paddle=True)) for i in tuple(ws.columns)[3][1::]]

    f = open("./results/question.txt",mode='w')

    for i in question:
        f.writelines(' '.join(i)+"\n")
    f.close()
    f = open("./results/answer.txt", mode="w")

    for i in answer:
        f.writelines(' '.join(i)+"\n")

    f.close()

def count_frequency():
    global question
    global answer
    filt_rules = re.compile(r'([0-9]|[a-zA-z]|\.|\s)+')
    stopwords = {'！',"？", "。","～","Ｂ",".",",",'，', '?'}
    question_words =  dict()
    answer_words = dict()
    
    for i in question:
        for j in i:
            if not filt_rules.match(j) and j not in stopwords:
                if j not in question_words:
                    question_words[j] = 1
                else:
                    question_words[j] += 1
    else:
        f=open("./results/question-words.txt", mode='w')
        for k, v in question_words.items():
            f.write("{0:<15}{1:<15}\n".format(k, v))
        f.close()

    for i in answer:
        for j in i:
            if not filt_rules.match(j) and j not in stopwords:
                if j not in answer:
                    answer_words[j] = 1
                else:
                    answer_words[j] += 1
    else:
        f=open("./results/answer-words.txt", mode='w')
        for k, v in answer_words.items():
            f.write("{0:<15}{1:<15}\n".format(k, v))
        f.close()

    # print(question_words, answer_words)

if __name__ == '__main__':
    participle()
    count_frequency()
