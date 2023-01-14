### 创建环境
```
	conda create -n slu python=3.6
	source activate slu
	pip install torch==1.7.1
```
### 运行
在根目录下运行：
+ `python scripts/slu_baseline.py` 进行模型训练

+ `python scripts/slu_baseline.py --testing` 对本地已有模型进行测试

+ `python scripts/test.py` 运行测试脚本以对文件`data/test_unlabelled.json` 进行预测

### 代码说明

#### data：
+ lexicon：储存语义三元组的部分定义
+ development.json：测试集
+ ontology.json：语义三元组的定义，更详细的定义在lexicon中
+ test_unlabelled.json：不含语义标记的数据
+ test.json：`scripts/test.py`的输出结果
+ train.json：训练集

#### utils：
+ args.py：参数设定，包含`batch_size`，`learning_rate`，`max_epoch`等神经网络训练参数。
+ batch.py：Batch类，训练批次
+ evaluator.py：Evaluator类，评估函数
+ example.py：Example类，样本
+ initialization.py：某些环境初始化
+ vocab.py：Vocab类，词表
+ word2vec.py：词和词向量间的转化

#### model：
+ slu_baseline_tagging.py：基线模型

#### scripts：
+ slu_baseline.py：基线模型的训练和测试脚本。默认为训练模式，传入参数`--testing`为测试模式。
+ test.py：测试脚本，用本地模型对无标签语料`data/test_unlabelled.json`进行预测，结果输出到`data/test.json`中。

其他：
+ model.bin：训练好的基线模型
+ word2vec-768.txt：词向量表
+ report.pdf：实验报告