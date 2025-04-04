# Author:刘越影
# Time: 2023-03-25 17:34
import torch


class myCRF:
    def __init__(self):
        self.scoreMap = {}  # 分数表
        self.UnigramTemplates = []  # 状态特征模板
        self.BigramTemplates = []  # 转移特征模板
        self.readTemplate("template.txt")  # 读取特征模板
        self.character_tagging("199801.txt", "train.txt")  # 原始数据集加工

    def readTemplate(self, template_file, debug=False):
        '''
        读取特征模板
        '''
        def parse_template_line(line):
            """解析模板行的辅助函数"""
            tmpList = []
            if line.find("/") > 0:
                left = line.split("/")[0].split("[")[-1].split(",")[0]
                right = line.split("/")[-1].split("[")[-1].split(",")[0]
                tmpList.append(int(left))
                tmpList.append(int(right))
            else:
                num = line.split("[")[-1].split(",")[0]
                tmpList.append(int(num))
            return tmpList

        with open(template_file, encoding='utf-8') as tempFile:
            switchFlag = False  # 先读Unigram，再读Bigram
            for line in tempFile:
                if line.find("Unigram") > 0 or line.find("Bigram") > 0:  # 读到'Unigram'或者'Bigram'
                    continue
                if switchFlag:
                    if len(line.strip()) == 0:
                        continue
                    self.BigramTemplates.append(parse_template_line(line))
                else:
                    if len(line.strip()) == 0:
                        switchFlag = True
                    else:
                        self.UnigramTemplates.append(parse_template_line(line))
        if debug:
            print(self.UnigramTemplates)
            print(self.BigramTemplates)

    def character_tagging(self, input_file, output_file):
        '''
        将原始数据集加工成4-tag形式
        '''
        with open(input_file, 'r', encoding='gbk') as input_data, \
             open(output_file, 'w', encoding='utf-8') as output_data:
            for line in input_data:
                word_list = line.strip().split(" ")
                for word in word_list[1:]:
                    words = word.split("/")
                    if len(words) >= 2:
                        xz = words[1]
                        word = words[0]
                        if len(word) == 1:
                            output_data.write(word + "\tS\n")
                        else:
                            output_data.write(word[0] + "\tB\n")
                            for w in word[1:-1]:
                                output_data.write(w + "\tM\n")
                            output_data.write(word[-1] + "\tE\n")
                output_data.write("\n")

    def getTrainData(self, train_file):
        '''
        读取数据集
        '''
        sentences = []
        results = []
        tempFile = open(train_file, encoding='utf-8')
        sentence = ""
        result = ""
        for line in tempFile:
            line = line.strip()
            if line == "":
                if sentence == "" or result == "":
                    pass
                else:
                    sentences.append(sentence)
                    results.append(result)
                sentence = ""
                result = ""
            else:
                data = line.split("\t")
                sentence += data[0]
                result += data[1]
        return [sentences, results]

    def makeKey(self, template, identity, sentence, pos, statusCovered, debug=False):
        '''
        模板标注函数：找出一句句子中，给定的模板下的某字符某标志相关的特征（BMES）
        :param template: 给定特征模板
        :param identity: 模板序号
        :param sentence: 标注句子
        :param pos: 当点位置
        :param statusCovered: 状态标注
        :param debug: 调试用
        :return: 标注结果
        '''
        result = ""
        result += identity
        for i in template:
            index = pos + i
            if index < 0 or index >= len(sentence):
                result += " "
            else:
                result += sentence[index]
        result += "/"
        result += statusCovered
        if (debug == True):
            print(result)
        return result

    def getUnigramScore(self, sentence, thisPos, thisStatus):
        '''
        获得给定词和标志的状态特征分数和
        :param sentence: 句子
        :param thisPos: 当前位置
        :param thisStatus: 当前标志
        :return: 得分
        '''
        unigramScore = 0
        for i, template in enumerate(self.UnigramTemplates):
            key = self.makeKey(template, str(i), sentence, thisPos, thisStatus)
            if key in self.scoreMap:
                feature_value, weight = self.scoreMap[key]
                unigramScore += feature_value * weight  # 特征值乘以权重
        return unigramScore

    def getBigramScore(self, sentence, thisPos, preStatus, thisStatus):
        '''
        获得给定词和标志的转移特征分数和
        :param sentence: 句子
        :param thisPos: 当前位置
        :param preStatus: 上一个特征
        :param thisStatus: 当前特征
        :return: 得分
        '''
        bigramScore = 0
        for i, template in enumerate(self.BigramTemplates):
            key = self.makeKey(template, str(i), sentence, thisPos, preStatus + thisStatus)
            if key in self.scoreMap:
                feature_value, weight = self.scoreMap[key]
                bigramScore += feature_value * weight  # 特征值乘以权重
        return bigramScore

    def num2Tag(self, number):
        '''
        将数字转为对应标志
        :param number: 数字
        :return: 标志
        '''
        if number == 0:
            return "B"
        elif number == 1:
            return "M"
        elif number == 2:
            return "E"
        elif number == 3:
            return "S"
        else:
            return None

    def tag2Num(self, status):
        '''
        将标志转为对应数字
        :param status: 标志
        :return: 数字
        '''
        if status == "B":
            return 0
        elif status == "M":
            return 1
        elif status == "E":
            return 2
        elif status == "S":
            return 3
        else:
            return -1

    def getMaxIndex(self, lst):
        '''
        获得最大值对应的索引
        '''
        return max(range(len(lst)), key=lst.__getitem__)

    def getDuplicate(self, s1, s2):
        '''
        状态序列里，正确的状态的个数
        '''
        return sum(1 for a, b in zip(s1, s2) if a == b)

    def getWrongNum(self, sentence, realRes):
        '''
        正确率计算函数
        :param sentence: 句子
        :param realRes: 正确解
        :return: 错误个数
        '''
        myRes = self.Viterbi(sentence)  # 我的解
        lens = len(sentence)
        wrongNum = 0
        for i in range(0, lens):
            myResI = myRes[i]  # 我的解
            theoryResI = realRes[i]  # 理论解
            if myResI != theoryResI:
                wrongNum += 1
        return wrongNum

    def setScoreMap(self, sentence, realRes, debug=False):
        '''
        建立状态特征和转移特征的特征矩阵，并依据结果为每个元素打分
        :param sentence: 句子
        :param realRes: 正确解
        :param debug: 调试用
        :return:
        '''
        myRes = self.Viterbi(sentence)  # 我的解
        learning_rate = 0.1  # 学习率
        for word in range(len(sentence)):  # 遍历整个句子
            myResI = myRes[word]  # 我的解
            theoryResI = realRes[word]  # 理论解
            if myResI != theoryResI:  # 如果和理论值不同
                # 更新Unigram特征
                for i, template in enumerate(self.UnigramTemplates):
                    myKey = self.makeKey(template, str(i), sentence, word, myResI)
                    theoryKey = self.makeKey(template, str(i), sentence, word, theoryResI)
                    # 更新我的解的权重
                    if myKey not in self.scoreMap:
                        self.scoreMap[myKey] = [1, -learning_rate]
                    else:
                        self.scoreMap[myKey][1] -= learning_rate
                    # 更新理论解的权重
                    if theoryKey not in self.scoreMap:
                        self.scoreMap[theoryKey] = [1, learning_rate]
                    else:
                        self.scoreMap[theoryKey][1] += learning_rate

                # 更新Bigram特征
                for i, template in enumerate(self.BigramTemplates):
                    if word == 0:
                        myKey = self.makeKey(template, str(i), sentence, word, " " + myResI)
                        theoryKey = self.makeKey(template, str(i), sentence, word, " " + theoryResI)
                    else:
                        myKey = self.makeKey(template, str(i), sentence, word, myRes[word - 1:word + 1])
                        theoryKey = self.makeKey(template, str(i), sentence, word, realRes[word - 1:word + 1])
                    # 更新我的解的权重
                    if myKey not in self.scoreMap:
                        self.scoreMap[myKey] = [1, -learning_rate]
                    else:
                        self.scoreMap[myKey][1] -= learning_rate
                    # 更新理论解的权重
                    if theoryKey not in self.scoreMap:
                        self.scoreMap[theoryKey] = [1, learning_rate]
                    else:
                        self.scoreMap[theoryKey][1] += learning_rate

    def Viterbi(self, sentence):
        '''
        结合scoremap使用维特比算法，先找到局部最优，记录节点，最后回溯得到路径。
        :param sentence: 句子
        :return: 路径
        '''
        lens = len(sentence)
        statusFrom = [[""] * lens, [""] * lens, [""] * lens, [""] * lens]  # B,M,E,S
        maxScore = [[0] * lens, [0] * lens, [0] * lens, [0] * lens]  # 4条路
        for word in range(0, lens):
            for stateNum in range(0, 4):
                thisStatus = self.num2Tag(stateNum)
                # 第一个词，状态特征加转移特征
                if word == 0:
                    uniScore = self.getUnigramScore(sentence, 0, thisStatus)
                    biScore = self.getBigramScore(sentence, 0, " ", thisStatus)
                    maxScore[stateNum][0] = uniScore + biScore
                    statusFrom[stateNum][0] = None
                else:
                    # 前面的所有路径到当前节点路径的所有得分之和
                    scores = [0] * 4
                    for i in range(0, 4):
                        preStatus = self.num2Tag(i)  # 记录前一节点
                        transScore = maxScore[i][word - 1]  # 到前一节点的路径和
                        uniScore = self.getUnigramScore(sentence, word, thisStatus)  # 状态特征分数
                        biScore = self.getBigramScore(sentence, word, preStatus, thisStatus)  # 转移特征分数
                        scores[i] = transScore + uniScore + biScore  # 当前节点分数
                    maxIndex = self.getMaxIndex(scores)  # 找到最大分数
                    maxScore[stateNum][word] = scores[maxIndex]  # 最大分数记录
                    statusFrom[stateNum][word] = self.num2Tag(maxIndex)  # 最大分数对应节点记录
        resBuf = [""] * lens
        scoreBuf = [0] * 4
        if lens > 0:
            for i in range(0, 4):
                scoreBuf[i] = maxScore[i][lens - 1]  # 最后一个字的各个标志最大分数
            resBuf[lens - 1] = self.num2Tag(self.getMaxIndex(scoreBuf))  # 最后一个字最大分数对应标志
        for backIndex in range(lens - 2, -1, -1):
            resBuf[backIndex] = statusFrom[self.tag2Num(resBuf[backIndex + 1])][backIndex + 1]  # 回溯路径
        res = "".join(resBuf)  # 输出路径
        return res

    def myTrain(self, data, model_path, epochnum=3):
        '''
        训练函数
        :param data: 训练数据
        :param model_path: 模型参数保存路径
        :param epochnum: 训练批次
        :return:
        '''
        sentences, results = self.getTrainData(data)  # 读取数据集
        whole = len(sentences)  # 句子数量
        trainNum = int(whole * 0.8)  # 选前80%句子作为训练集
        for epoch in range(1, epochnum):  # 训练次数
            wrongNum = 0
            totalTest = 0  # 记录字符数
            for i in range(0, trainNum):
                sentence = sentences[i]
                totalTest += len(sentence)
                result = results[i]
                self.setScoreMap(sentence, result)  # 训练的关键，计算scoreMap
                wrongNum += self.getWrongNum(sentence, result)  # 计算错误的点数
                if i % 1000 == 0:  # 每1000个句子打印一次
                    correctNum = totalTest - wrongNum  # 正确点数
                    print("epoch" + str(epoch) + f" {i}/{trainNum} " + ":准确率" + str(float(correctNum / totalTest)))
            correctNum = totalTest - wrongNum  # 正确点数
            print("epoch" + str(epoch) + ":准确率" + str(float(correctNum / totalTest)))  # 计算正确率
            total = 0
            correct = 0
            # 测试集为后20%
            for i in range(trainNum, whole):
                sentence = sentences[i]
                total += len(sentence)
                result = results[i]
                myRes = self.Viterbi(sentence)
                correct += self.getDuplicate(result, myRes)
            accuracy = float(correct / total)  # 计算测试集正确率
            print("accuracy" + str(accuracy))
            torch.save(
                {
                    'scoreMap': self.scoreMap,
                    'BigramTemplates': self.BigramTemplates,
                    'UnigramTemplates': self.UnigramTemplates
                },
                model_path
            )

    def predict(self, sentence, parameter):
        '''
        "解码"函数
        :param sentence: 句子
        :param parameter: 参数
        :return:分词结果
        '''
        # global count
        retsentence = []
        state = []
        # count += 1
        # print(count)
        self.scoreMap = parameter['scoreMap']
        self.UnigramTemplates = parameter['UnigramTemplates']
        self.BigramTemplates = parameter['BigramTemplates']
        retsentence = self.Viterbi(sentence)
        if retsentence[-1] == 'B' or retsentence[-1] == 'M':  # 最后一个字状态不是'S'或'E'则修改
            if retsentence[-2] == 'B'or retsentence[-2] == 'M':
                retsentence[-1] = 'E'
            else:
                retsentence[-1] = 'S'

        # 开始对该行分词
        curLine = ''
        # 遍历该行每一个字
        for i in range(len(sentence)):
            # 在列表中放入该字
            curLine += sentence[i]
            # 如果该字是S->单个词  或  E->结尾词 ，则在该字后面加上分隔符 |
            # 此外如果改行的最后一个字了，也就不需要加 |
            if (retsentence[i] == 'S'or retsentence[i] == 'E') and i != (len(sentence) - 1):
                curLine += '|'
        # 在返回列表中添加分词后的该行
        state.append(curLine)
        return state


if __name__ == '__main__':
    model = myCRF()
    model.myTrain("train.txt", "CRF-dataSet.model", epochnum=2)
    parameter = torch.load("CRF-dataSet.model")
    print(model.predict("打工人打工魂，打工就是人上人。", parameter))