import torch


class myCRF:
    def __init__(self):
        self.scoreMap = {}  # 分数表
        self.UnigramTemplates = []  # 状态特征模板
        self.BigramTemplates = []  # 转移特征模板
        self.readTemplate("template.txt")  # 读取特征模板
        self.character_tagging("train_set.txt", "train_2.txt")  # 原始数据集加工

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
        with open(input_file, 'r', encoding='utf-8') as input_data, \
             open(output_file, 'w', encoding='utf-8') as output_data:
            for line in input_data:
                word_list = line.strip().split(":")
                word_list = word_list[-1].strip().split(" ")
                for word in word_list[:]:
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

    def train_with_gradient_descent(self, data, model_path, epochnum=3, learning_rate=0.01):
        '''
        使用梯度下降的训练方式
        :param data: 训练数据
        :param model_path: 模型参数保存路径
        :param epochnum: 训练批次
        :param learning_rate: 学习率
        :return:
        '''
        sentences, results = self.getTrainData(data)  # 读取数据集
        whole = len(sentences)  # 句子数量
        trainNum = int(whole * 0.8)  # 选前80%句子作为训练集

        # 初始化权重
        for epoch in range(1, epochnum + 1):
            total_loss = 0
            for i in range(0, trainNum):
                sentence = sentences[i]
                result = results[i]
                myRes = self.Viterbi(sentence)  # 当前预测结果
                for word in range(len(sentence)):
                    myResI = myRes[word]
                    theoryResI = result[word]
                    if myResI != theoryResI:
                        # 更新Unigram特征
                        for j, template in enumerate(self.UnigramTemplates):
                            myKey = self.makeKey(template, str(j), sentence, word, myResI)
                            theoryKey = self.makeKey(template, str(j), sentence, word, theoryResI)
                            if myKey not in self.scoreMap:
                                self.scoreMap[myKey] = [1, -learning_rate]
                            else:
                                self.scoreMap[myKey][1] -= learning_rate
                            if theoryKey not in self.scoreMap:
                                self.scoreMap[theoryKey] = [1, learning_rate]
                            else:
                                self.scoreMap[theoryKey][1] += learning_rate

                        # 更新Bigram特征
                        for j, template in enumerate(self.BigramTemplates):
                            if word == 0:
                                myKey = self.makeKey(template, str(j), sentence, word, " " + myResI)
                                theoryKey = self.makeKey(template, str(j), sentence, word, " " + theoryResI)
                            else:
                                myKey = self.makeKey(template, str(j), sentence, word, myRes[word - 1:word + 1])
                                theoryKey = self.makeKey(template, str(j), sentence, word, result[word - 1:word + 1])
                            if myKey not in self.scoreMap:
                                self.scoreMap[myKey] = [1, -learning_rate]
                            else:
                                self.scoreMap[myKey][1] -= learning_rate
                            if theoryKey not in self.scoreMap:
                                self.scoreMap[theoryKey] = [1, learning_rate]
                            else:
                                self.scoreMap[theoryKey][1] += learning_rate

                # 计算损失
                wrongNum = self.getWrongNum(sentence, result)
                total_loss += wrongNum

            print(f"Epoch {epoch}/{epochnum}, Loss: {total_loss}")
            torch.save(
                {
                    'scoreMap': self.scoreMap,
                    'BigramTemplates': self.BigramTemplates,
                    'UnigramTemplates': self.UnigramTemplates
                },
                model_path
            )

    def myTrain(self, data, model_path, epochnum=3, use_gradient_descent=False):
        '''
        训练函数
        :param data: 训练数据
        :param model_path: 模型参数保存路径
        :param epochnum: 训练批次
        :param use_gradient_descent: 是否使用梯度下降训练方式
        :return:
        '''
        if use_gradient_descent:
            self.train_with_gradient_descent(data, model_path, epochnum)
            return

        # 如果模型文件存在，加载权重
        try:
            checkpoint = torch.load(model_path)
            self.scoreMap = checkpoint['scoreMap']
            self.BigramTemplates = checkpoint['BigramTemplates']
            self.UnigramTemplates = checkpoint['UnigramTemplates']
            print(f"Loaded model weights from {model_path}")
        except FileNotFoundError:
            print(f"No existing model found at {model_path}, starting training from scratch.")

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
                if i % 4000 == 0:  # 每1000个句子打印一次
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
    # model.myTrain("train.txt", "CRF-dataSet copy.model", epochnum=7)
    parameter = torch.load("CRF-dataSet copy.model")
    # test_file_path = "test.txt"
    # sentences = []
    # with open(test_file_path, "r", encoding="utf-8") as f:
    #     for line in f:
    #         sentences.append(line.strip())
    #         print(model.predict(line.strip(), parameter))
    
    # s = "明明明明明白白白喜欢他可是他就是不说。"
    # s = "人要是行，干一行行一行，一行行行行行，行行行干哪行都行。要是不行，干一行不行一行，一行不行行行不行，行行不行干哪行都不行。"
    # s = "重庆市长江边的景色迷人，让人流连忘返。"
    # s = "他说的确实在理，这里没有小明要的乒乓球拍，因为乒乓球拍卖完了。"
    # s = "校长说：校服上除了校徽别别别的，让你们别别别的别别别的你非别别的！"
    # s = "北京大学生在北京大学学习期间，积极参与研究生命起源的跨学科项目，致力于揭示生命演化的奥秘。"
    ss = ["中文信息处理课程由同济大学卫老师讲授。", "中国的历史源远流长，有着上下五千年的历史。", "当代大学生在追求梦想的同时，也面临着现实生活中不断变化的竞争和压力。", "两千里的河堤，已经完全支离破碎了，许多地方被敌伪挖成了封锁沟，许多地方被农民改成了耕地。再加上风吹雨打，使许多段河堤连痕迹都没有了。", "最高人民法院认定该区块链金融合约因违反反洗钱法相关条款而无效。", "计算机科学与技术学院的研究生们在国际知名期刊上发表了大量高水平论文，获得广泛认可。"]
    for s in ss:
        print(model.predict(s, parameter))