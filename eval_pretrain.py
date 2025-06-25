import os
import json
import jittor as jt
from model import ModelArgs, Transformer
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import numpy as np

jt.flags.use_cuda = 1 
jt.flags.log_silent = True  

# 配置参数
out_dir = 'out'  
start = ""  # 起始文本
num_samples = 1  # 生成的样本数量
max_new_tokens = 100  # 每个样本生成的最大token数
temperature = 1.0  # 温度参数，控制随机性
top_k = 30  # 只考虑top_k个最可能的token
seed = 1337  # 随机种子

# 模型参数配置
max_seq_len = 256
dim = 512
n_layers = 8
n_heads = 8
multiple_of = 32
dropout = 0.0 

model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_heads,
    vocab_size=64793,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
)

# 设置随机种子
jt.set_global_seed(seed)

# 初始化模型
gptconf = ModelArgs(**model_args)
model = Transformer(gptconf)

# 加载模型权重
ckpt_path = 'out/pretrain/epoch_bigger0.pkl'
model.load_parameters(jt.load(ckpt_path))

# 设置为评估模式
model.eval()

# 加载tokenizer
tokenizer = ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')

# 测试数据
data = [
    {"question": "高考是，"},
    {"question": "樟树是，"},
    {"question": "百度是，"},
]

# 生成文本
ans_lst = []
for p in data:
    # 准备输入
    prompt = p['question']
    x = tokenizer.encode(prompt, add_special_tokens=False)
    x = jt.array(x).int().unsqueeze(0)  # 添加batch维度
    
    # 使用Jittor的no_grad上下文
    with jt.no_grad():
        # 生成文本
        y = model.generate(x, 2, max_new_tokens, temperature=temperature, top_k=top_k)
        
        # 解码生成的文本
        generated_ids = y[0].numpy().tolist()
        answer = tokenizer.decode(generated_ids)
        answer = answer.replace(prompt, '')  # 移除原始提示
        
        ans_lst.append(answer)
        print('[prompt]:', prompt)
        print('[answer]:', answer)
        print('---------------')