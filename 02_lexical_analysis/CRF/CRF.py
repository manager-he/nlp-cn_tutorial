import re
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import pickle

# 数据预处理
class CorpusProcess(object):
    def __init__(self, data_dir='data/'):
        self.train_corpus_path = data_dir + "train_data.txt"  # 分词训练数据路径
        self.process_corpus_path = data_dir + "processed_data.txt"
        self._maps = {'B': 'B', 'M': 'M', 'E': 'E', 'S': 'S'}

    def read_corpus_from_file(self, file_path):
        f = open(file_path, 'r', encoding='utf-8')
        lines = f.readlines()
        f.close()
        return lines

    def write_corpus_to_file(self, data, file_path):
        f = open(file_path, 'wb')
        f.write(data)
        f.close()

    def pre_process(self):
        lines = self.read_corpus_from_file(self.train_corpus_path)
        new_lines = []
        for line in lines:
            words = line.strip().split(' ')
            pro_words = self.process_words(words)
            new_lines.append(' '.join(pro_words))
        self.write_corpus_to_file(data='\n'.join(new_lines).encode('utf-8'), file_path=self.process_corpus_path)

    def process_words(self, words):
        pro_words = []
        for word in words:
            if len(word) == 1:
                pro_words.append(word + '/S')
            elif len(word) == 2:
                pro_words.append(word[0] + '/B ' + word[1] + '/E')
            else:
                pro_words.append(word[0] + '/B ' + ' '.join([char + '/M' for char in word[1:-1]]) + ' ' + word[-1] + '/E')
        return pro_words

    def initialize(self):
        lines = self.read_corpus_from_file(self.process_corpus_path)
        words_list = [line.strip().split(' ') for line in lines if line.strip()]
        del lines
        self.init_sequence(words_list)

    def init_sequence(self, words_list):
        words_seq = [[word.split('/')[0] for word in words] for words in words_list]
        tag_seq = [[word.split('/')[1] for word in words] for words in words_list]
        self.word_seq = [['<BOS>'] + [w for word in word_seq for w in word] + ['<EOS>'] for word_seq in words_seq]
        self.tag_seq = [[t for tag in tag_seq for t in tag] for tag_seq in tag_seq]

    def extract_feature(self, word_grams):
        features, feature_list = [], []
        for index in range(len(word_grams)):
            for i in range(len(word_grams[index])):
                word_gram = word_grams[index][i]
                feature = {'w-1': word_gram[0], 'w': word_gram[1], 'w+1': word_gram[2],
                           'w-1:w': word_gram[0] + word_gram[1], 'w:w+1': word_gram[1] + word_gram[2],
                           'bias': 1.0}
                feature_list.append(feature)
            features.append(feature_list)
            feature_list = []
        return features

    def segment_by_window(self, words_list=None, window=3):
        words = []
        begin, end = 0, window
        for _ in range(1, len(words_list)):
            if end > len(words_list): break
            words.append(words_list[begin:end])
            begin = begin + 1
            end = end + 1
        return words

    def generator(self):
        word_grams = [self.segment_by_window(word_list) for word_list in self.word_seq]
        features = self.extract_feature(word_grams)
        return features, self.tag_seq

# CRF分词器
class CRF_Seg(object):
    def __init__(self, data_dir='data/'):
        self.algorithm = "lbfgs"
        self.c1 = "0.1"
        self.c2 = "0.1"
        self.max_iterations = 100
        self.model_path = data_dir + "seg_model.pkl"
        self.corpus = CorpusProcess(data_dir)  # Corpus 实例
        self.corpus.pre_process()  # 语料预处理
        self.corpus.initialize()  # 初始化语料
        self.model = None

    def initialize_model(self):
        algorithm = self.algorithm
        c1 = float(self.c1)
        c2 = float(self.c2)
        max_iterations = int(self.max_iterations)
        self.model = sklearn_crfsuite.CRF(algorithm=algorithm, c1=c1, c2=c2,
                                          max_iterations=max_iterations, all_possible_transitions=True)

    def train(self):
        self.initialize_model()
        x, y = self.corpus.generator()
        x_train, y_train = x[500:], y[500:]
        x_test, y_test = x[:500], y[:500]
        self.model.fit(x_train, y_train)
        labels = list(self.model.classes_)
        labels.remove('O')
        y_predict = self.model.predict(x_test)
        metrics.flat_f1_score(y_test, y_predict, average='weighted', labels=labels)
        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
        print(metrics.flat_classification_report(y_test, y_predict, labels=sorted_labels, digits=3))
        self.save_model()

    def predict(self, sentence):
        self.load_model()
        u_sent = sentence
        word_lists = [['<BOS>'] + [c for c in u_sent] + ['<EOS>']]
        word_grams = [self.corpus.segment_by_window(word_list) for word_list in word_lists]
        features = self.corpus.extract_feature(word_grams)
        y_predict = self.model.predict(features)
        words = []
        start = -1
        for index in range(len(y_predict[0])):
            if y_predict[0][index] == 'B':
                if start != -1:
                    words.append(u_sent[start:index])
                start = index
            elif y_predict[0][index] == 'S':
                if start != -1:
                    words.append(u_sent[start:index])
                words.append(u_sent[index])
                start = -1
            elif y_predict[0][index] == 'E':
                words.append(u_sent[start:index+1])
                start = -1
        return words

    def load_model(self):
        self.model = pickle.load(open(self.model_path, 'rb'))

    def save_model(self):
        pickle.dump(self.model, open(self.model_path, 'wb'))

# 使用示例
seg = CRF_Seg(data_dir='data/')
seg.train()
result = seg.predict("长春市长春节讲话")
print(result)