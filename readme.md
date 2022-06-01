## Step1. 将仓库克隆到本地并安装依赖

```bash
git clone https://github.com/RileyYe/Soochow-Univ-NLP-Experiments.git
cd ./Soochow-Univ-NLP-Experiments
pip install -r ./requirements.txt 
```

## Step2.1 Exp1-中文文本分词与词频统计
### 运行方式
```bash
cd ./Exp1-中文文本分词与词频统计
python3 ./main.py # Linux环境
```

## Step2.2 Exp2-借助开源TensorFlow工具的词向量训练
### 运行方式
```bash
cd ./Exp2-借助开源TensorFlow工具的词向量训练
tar -xf ./data.tar.gz
python3 ./main.py # Linux环境
```
### 训练方式
```bash
cd ./Exp2-借助开源TensorFlow工具的词向量训练
python3 ./dump_conf_dist.py
python3 ./dump_top5.py
```