import unittest
import torch
from sampleCRF import myCRF

class TestMyCRF(unittest.TestCase):
    def setUp(self):
        self.model = myCRF()
        # 初始化特征模板和分数表
        self.model.UnigramTemplates = [[0], [1]]
        self.model.BigramTemplates = [[0, 1]]
        self.model.scoreMap = {
            "0a/B": 1.0, "0b/M": 0.5, "1c/E": 0.8,
            "0 /B": 0.2, "0B/M": 0.3, "0M/E": 0.4
        }

    def test_compute_neg_log_likelihood(self):
        sentence = "abc"
        tags = "BME"
        # 计算负对数似然值
        neg_log_likelihood = self.model.compute_neg_log_likelihood(sentence, tags, debug=True)
        # 手动计算期望值
        gold_score = self.model.compute_sentence_score(sentence, tags)
        partition_function = self.model.compute_partition_function(sentence)
        expected_value = partition_function - gold_score
        # 检查结果是否正确
        self.assertAlmostEqual(neg_log_likelihood.item(), expected_value.item(), places=5)

if __name__ == "__main__":
    unittest.main()
