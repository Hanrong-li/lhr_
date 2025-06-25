from collections import Counter
from typing import Union
from dataclasses import make_dataclass, field
from transformers import T5Config
import ctypes
import os
import platform
import re

import jittor as jt
from datasketch import MinHash, MinHashLSH
from collections import defaultdict
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers import TrainingArguments, TrainerCallback

import jieba
import pandas as pd
from simhash import Simhash, SimhashIndex

from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import ujson

# 结束标点符号
END_PUN = set(".。!！）)》}】?？\"”")

class MyTrainerCallback(TrainerCallback):
    log_cnt = 0
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        '''
        在打印 n 次日志后清除缓存，适合低显存设备，能防止OOM
        '''
        self.log_cnt += 1
        if self.log_cnt % 2 == 0:
            jt.clean()  # 使用Jittor的内存清理函数
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        '''
        在 on_epoch_end 时保存一次模型。
        '''
        control.should_save = True
        return control


# 保留中文和英文、下划线，不要标点符号
NON_CHAR = re.compile("[^[\u4E00-\u9FA5|A-Za-z_0-9]")

def _get_doc_mini_hash(doc, num_perm: int) -> MinHash:
    '''
    获取一段文本的mini hash
    '''
    mini_hash = MinHash(num_perm=num_perm)
    for s in doc:
        mini_hash.update(s.encode('utf-8'))
    return mini_hash

class DropDatasetDuplicate:

    def __init__(self,  threshold: float=0.85, num_perm: int=256) -> None:
        self.similar_index_cluster = defaultdict(set)
        self.data_lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm

    def add_doc(self, index, doc: str):
        doc = ''.join(NON_CHAR.split(doc))
        doc_hash = _get_doc_mini_hash(doc, self.num_perm)
        close_duplicates = self.data_lsh.query(doc_hash)

        self.data_lsh.insert(index, doc_hash)

        if len(close_duplicates) > 0:
            min_idx= min(close_duplicates)
            self.similar_index_cluster[min_idx].add(index)
    
    def get_duplicate_indexs(self):
        need_to_remove_idx = set()
        
        for key_idx in self.similar_index_cluster.keys():
            need_to_remove_idx |= self.similar_index_cluster[key_idx]

        return need_to_remove_idx

class DropDatasetDuplicate_SimHash:
    def __init__(self, threshold: int = 3, f: int = 64) -> None:
        self.database = {}
        self.dupcount = 0
        self.index = SimhashIndex([], k=threshold, f=f)
        self.threshold = threshold
        self.f = f

    def get_features(self, s: str):
        width = 3
        s = s.lower()
        s = re.sub(r'[^\w]+', '', s)
        return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]

    def add_doc(self, index, doc: str):
        if index == 0:
            self.database[index] = doc
            self.index.add(str(index), Simhash(self.get_features(doc), f=self.f))
        else:
            s1 = Simhash(self.get_features(doc), f=self.f)
            if self.index.get_near_dups(s1) == []:
                self.database[index] = doc
                self.index.add(str(index), s1)
            else:
                self.dupcount += 1

def f1_p_r_compute(spo_list_pred: list, spo_list_true: list, repair: bool=False):
    if repair:
        spo_list_pred = repair_song_album_list(spo_list_pred)
        spo_list_true = repair_song_album_list(spo_list_true)

    TP = 1e-10
    TP_FP = 1e-10
    TP_FN = 1e-10

    for i in range(len(spo_list_true)):
        pred_set = set(spo_list_pred[i])
        true_set = set(spo_list_true[i])
        pred_true_set = pred_set & true_set

        TP += len(pred_true_set)
        TP_FP += len(pred_set)
        TP_FN += len(true_set)

    p = TP / TP_FP
    r = TP / TP_FN
    f1 = (2 * p * r) / (p + r)
    
    return f1, p, r

def fixed_response(item: str) -> str:
    if len(item) <= 1: return item
    if item[-1] in END_PUN: return item

    n = len(item)
    i = n - 1
    while i > 0 and item[i] not in END_PUN:
        i -= 1

    return ''.join(item[0: i + 1])

def fixed_space(sentence: str)->str:
    n = len(sentence)
    new_sentence = []
    i = 0
    while i < n:
        word =  sentence[i]
        if word != ' ':
            new_sentence.append(word)
        elif i + 1 < n and sentence[i + 1] == ' ':
            new_sentence.append(word)
            i += 1
        i += 1

    return ''.join(new_sentence)

def get_free_space_of_disk(folder: str='./') -> float:
    res_val = 0.0
    if platform.system() == 'Windows':
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(folder), None, None, ctypes.pointer(free_bytes))
        res_val = free_bytes.value 
    else:
        st = os.statvfs(folder)
        res_val = st.f_bavail * st.f_frsize
    
    return res_val / (1024 ** 3)

def my_average(arry_list) -> float:
    if len(arry_list) == 0: return 0.0
    return np.average(arry_list)

def json_to_dataclass(json_file: str, class_name: str='Config') -> type:
    json_dict = {}
    with open(json_file, 'r', encoding='utf-8') as f:
        json_dict = ujson.load(f)

    fields_list = []
    for k, v in json_dict.items():
        fields_list.append( (k, type(v), field(default=v)) )
    
    data_class = make_dataclass(cls_name=class_name, fields=fields_list)
    return data_class

def get_path_of_suffix_files(root: str, suffix: str, with_create_time: bool=False) -> list:
    suffix_files = []
    for root, _, files in os.walk(root):
        for file in files:
            if file.endswith(suffix):
                full_path = os.path.join(root, file)
                if with_create_time:
                    suffix_files.append( (full_path, os.path.getctime(full_path)) )
                else:
                    suffix_files.append(full_path)
                            
    return suffix_files

def get_bleu4_score(reference, outputs, n_gram=4):
    weights = np.ones(n_gram) * (1.0 / n_gram)
    outputs_len, reference_len = len(outputs), len(reference)

    if not isinstance(reference, list):
        reference = list(reference)
    if not isinstance(outputs, list):
        outputs = list(outputs)

    outputs_counter = extract_Ngram(outputs, n_gram=n_gram)
    reference_counter = extract_Ngram(reference, n_gram=n_gram)

    ngram_counter_clip = outputs_counter & reference_counter

    clip_counter = np.zeros(n_gram)
    output_ngram_counter = np.zeros(n_gram)

    for (key, ngram), cnt in ngram_counter_clip.items():
        clip_counter[ngram - 1] += cnt 
    
    for (key, ngram), cnt in outputs_counter.items():
        output_ngram_counter[ngram - 1] += cnt
    
    if np.min(clip_counter) == 0.0:
        return np.array(0.0)

    precision_scores = clip_counter / output_ngram_counter
    log_precision_scores = weights * np.log(precision_scores)
    
    geometric_mean = np.exp(np.sum(log_precision_scores))
    brevity_penalty = np.exp(1 - (reference_len / outputs_len))

    bleu = brevity_penalty * geometric_mean
    return bleu

def extract_Ngram(words_list, n_gram):
    n = len(words_list)
    ngram_counter = Counter()

    for i in range(1, n_gram + 1):
        for j in range(n - i + 1):
            key = ' '.join(words_list[j: j + i])
            ngram_counter[(key, i)] += 1

    return ngram_counter

def save_model_config(config_dict, file):
    with open(file, 'w', encoding='utf-8') as f:
        ujson.dump(config_dict, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    ref = '抱歉，我不知道ABB代表什么意思'
    out = '我不明白ABB是什么意思'
    b1 = sentence_bleu([list(out)], list(ref),  weights=(0.25, 0.25, 0.25, 0.25))
    print(b1)
    b2 = get_bleu4_score(out, ref)
    print(b2)

    candidate_corpus = ['i', 'have', 'a', 'pen', 'on', 'my', 'desk', 'a', 'b', 'c', 'd','f','f']
    reference_corpus = ['there', 'is', 'a', 'pen', 'on', 'my', 'desk', 'a', 'b', 'd', 'd', 'fd']
    
    print('----')
    print(sentence_bleu([reference_corpus], candidate_corpus,  weights=(0.25, 0.25, 0.25, 0.25)))
    print(get_bleu4_score(reference_corpus, candidate_corpus))