import torch


class myCRF:
    def __init__(self):
        self.scoreMap = {}  # 特征分数表
        self.UnigramTemplates = []  # 状态特征模板
        self.BigramTemplates = []  # 转移特征模板
        self.readTemplate("template.txt")  # 读取特征模板
        self.character_tagging("train_set.txt", "train_2.txt")  # 加工原始数据集

    def readTemplate(self, template_file, debug=False):
        '''
        读取特征模板文件并解析
        '''
        def parse_template_line(line):
            """解析模板行"""
            tmpList = []
            if "/" in line:
                left, right = [int(x.split("[")[-1].split(",")[0]) for x in line.split("/")]
                tmpList.extend([left, right])
            else:
                tmpList.append(int(line.split("[")[-1].split(",")[0]))
            return tmpList

        with open(template_file, encoding='utf-8') as tempFile:
            switchFlag = False  # 切换标志，先读Unigram，再读Bigram
            for line in tempFile:
                line = line.strip()
                if "Unigram" in line or "Bigram" in line:
                    continue
                if switchFlag:
                    if line:
                        self.BigramTemplates.append(parse_template_line(line))
                else:
                    if not line:
                        switchFlag = True
                    else:
                        self.UnigramTemplates.append(parse_template_line(line))
        if debug:
            print("Unigram Templates:", self.UnigramTemplates)
            print("Bigram Templates:", self.BigramTemplates)

    def character_tagging(self, input_file, output_file):
        '''
        将原始数据集加工成4-tag形式（B, M, E, S）
        '''
        def write_tags(word, output_data):
            """根据词长度写入对应标签"""
            if len(word) == 1:
                output_data.write(f"{word}\tS\n")
            else:
                output_data.write(f"{word[0]}\tB\n")
                output_data.writelines(f"{char}\tM\n" for char in word[1:-1])
                output_data.write(f"{word[-1]}\tE\n")

        with open(input_file, 'r', encoding='utf-8') as input_data, \
             open(output_file, 'w', encoding='utf-8') as output_data:
            for line in input_data:
                word_list = line.strip().split(":")[-1].strip().split(" ")
                for word in word_list:
                    parts = word.split("/")
                    if len(parts) >= 2:
                        write_tags(parts[0], output_data)
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
        模板标注函数：生成特征键
        '''
        result = identity + ''.join(
            sentence[pos + i] if 0 <= pos + i < len(sentence) else " " for i in template
        ) + f"/{statusCovered}"
        if debug:
            print(result)
        return result

    def updateScoreMap(self, key, weight_delta):
        '''
        更新特征分数表
        '''
        if key not in self.scoreMap:
            self.scoreMap[key] = [1, weight_delta]
        else:
            self.scoreMap[key][1] += weight_delta

    def updateFeatures(self, sentence, word, myResI, theoryResI, learning_rate):
        '''
        更新Unigram和Bigram特征
        '''
        for i, template in enumerate(self.UnigramTemplates):
            myKey = self.makeKey(template, str(i), sentence, word, myResI)
            theoryKey = self.makeKey(template, str(i), sentence, word, theoryResI)
            self.updateScoreMap(myKey, -learning_rate)
            self.updateScoreMap(theoryKey, learning_rate)

        for i, template in enumerate(self.BigramTemplates):
            prev_my = " " + myResI if word == 0 else myResI[word - 1:word + 1]
            prev_theory = " " + theoryResI if word == 0 else theoryResI[word - 1:word + 1]
            myKey = self.makeKey(template, str(i), sentence, word, prev_my)
            theoryKey = self.makeKey(template, str(i), sentence, word, prev_theory)
            self.updateScoreMap(myKey, -learning_rate)
            self.updateScoreMap(theoryKey, learning_rate)

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
        for word in range(len(sentence)):
            if myRes[word] != realRes[word]:
                self.updateFeatures(sentence, word, myRes[word], realRes[word], learning_rate)

    def Viterbi(self, sentence):
        '''
        结合scoremap使用维特比算法，先找到局部最优，记录节点，最后回溯得到路径。
        :param sentence: 句子
        :return: 路径
        '''
        lens = len(sentence)
        maxScore = [[0] * lens for _ in range(4)]  # 4条路径
        statusFrom = [[""] * lens for _ in range(4)]  # B, M, E, S

        for word in range(lens):
            for stateNum in range(4):
                thisStatus = self.num2Tag(stateNum)
                if word == 0:
                    maxScore[stateNum][0] = self.getUnigramScore(sentence, 0, thisStatus) + \
                                            self.getBigramScore(sentence, 0, " ", thisStatus)
                    statusFrom[stateNum][0] = None
                else:
                    scores = [
                        maxScore[i][word - 1] +
                        self.getUnigramScore(sentence, word, thisStatus) +
                        self.getBigramScore(sentence, word, self.num2Tag(i), thisStatus)
                        for i in range(4)
                    ]
                    maxIndex = self.getMaxIndex(scores)
                    maxScore[stateNum][word] = scores[maxIndex]
                    statusFrom[stateNum][word] = self.num2Tag(maxIndex)

        resBuf = [""] * lens
        if lens > 0:
            resBuf[-1] = self.num2Tag(self.getMaxIndex([maxScore[i][-1] for i in range(4)]))
            for backIndex in range(lens - 2, -1, -1):
                resBuf[backIndex] = statusFrom[self.tag2Num(resBuf[backIndex + 1])][backIndex + 1]
        return "".join(resBuf)

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
                        self.updateFeatures(sentence, word, myResI, theoryResI, learning_rate)

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
        解码函数：根据输入句子和模型参数进行分词预测
        '''
        self.scoreMap = parameter['scoreMap']
        self.UnigramTemplates = parameter['UnigramTemplates']
        self.BigramTemplates = parameter['BigramTemplates']

        # 使用Viterbi算法预测状态序列
        predicted_states = self.Viterbi(sentence)

        # 修正最后一个字的状态
        if predicted_states[-1] in {'B', 'M'}:
            predicted_states[-1] = 'E' if predicted_states[-2] in {'B', 'M'} else 'S'

        # 根据状态序列进行分词
        segmented_sentence = []
        current_word = ''
        for i, char in enumerate(sentence):
            current_word += char
            if predicted_states[i] in {'S', 'E'} and i != len(sentence) - 1:
                current_word += '|'
        segmented_sentence.append(current_word)
        return segmented_sentence


if __name__ == '__main__':
    model = myCRF()
    
    # model.myTrain("train.txt", "CRF-dataSet final.model", epochnum=2)
    
    # 加载模型参数
    parameter = torch.load("CRF-dataSet final.model")

    # 示例句子
    test_sentences = [
        "中文信息处理课程由同济大学卫老师讲授。",
        "中国的历史源远流长，有着上下五千年的历史。",
        "当代大学生在追求梦想的同时，也面临着现实生活中不断变化的竞争和压力。",
        "两千里的河堤，已经完全支离破碎了，许多地方被敌伪挖成了封锁沟，许多地方被农民改成了耕地。再加上风吹雨打，使许多段河堤连痕迹都没有了。",
        "最高人民法院认定该区块链金融合约因违反反洗钱法相关条款而无效。",
        "计算机科学与技术学院的研究生们在国际知名期刊上发表了大量高水平论文，获得广泛认可。"
    ]

    # 输出分词结果
    for sentence in test_sentences:
        print(model.predict(sentence, parameter))