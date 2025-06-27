import ujson
import re
from os.path import dirname, abspath, exists, isdir
from os import remove, mkdir, walk
import time
from collections import defaultdict

from matplotlib import pyplot as plt
import codecs, csv
import pandas as pd 
import numpy as np
from rich import progress
from rich.table import Table
from rich.console import Console
from fastparquet import ParquetFile, write
import pyarrow.parquet as pq

import sys
sys.path.extend(['.','..'])

from logger import Logger
from functions import get_path_of_suffix_files, DropDatasetDuplicate, DropDatasetDuplicate_SimHash

log = Logger('data_process', save2file=True, file_name='./logs/raw_data_process.log')

punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：\n")
en_punctuation = ",().!;:"
zh_punctuation = "，（）。！；："

def delete_file(file: str)-> bool:
    '''
    询问删除文件
    '''
    if exists(file):
        ans = input('delete file: {} ? Yes (y) or No (n)'.format(file))
        ans = ans.lower()
        if ans in ('yes', 'y'):
            remove(file)
            print('deleted.')
            return True
    return False

def remove_duplicate_punctuation(sentence: str) -> str:
    '''
    删除句子中重复的标点符号、重复的空格，同时将换行变为特殊字符'\n'
    '''
    # 将空格（全角空格）替换为逗号, 可能会有重复的空客，下面删除重复标点会删除
    sentence = re.sub(' |　', '，', sentence) 

    ans = ''
    n = len(sentence)
    p = 0
    while p < n:
        ans += sentence[p]

        while p + 1 < n and sentence[p] in punctuation and sentence[p + 1] in punctuation:
            p += 1
        p += 1

    return ans

def convert_en_punctuation_to_zh_punct(sentence: str) -> str:
    '''
    将句子中的英文标点替换文中文标点
    '''
    n = len(zh_punctuation)
    for i in range(n):
        sentence = sentence.replace(en_punctuation[i], zh_punctuation[i])
    return sentence

def get_sentences_dice_similarity(st_a: str, st_b: str) -> float:
    '''
    获取两个句子的Dice相似度（Dice similarity）
    s(a, b) =  2 * len( set(a) & set(b) ) / (len(set(a)) + len(set(b)))
    '''
    set_a, set_b = set(st_a), set(st_b)
    total_len  = len(set_a) + len(set_b)
    
    if total_len == 0: return 0.0

    inter_set =  set_a & set_b
    
    return ( 2 * len(inter_set)) / total_len

def write_single_parquet_file(file_name: str, data_frame: pd.DataFrame) -> None:
    '''
    将dataframe写到单独的parquet file中
    '''
    append = False
    if exists(file_name):
        append = True 

    write(file_name, data_frame, compression='GZIP', append=append)
    

def merge_dataset_as_single_file(groups_cnt: int=50000, max_len: int=512, min_len: int=3, cut_max_len: bool=False) -> None:
    '''
    将多个数据集合并为一个数据集
    '''
    from_parquet_files = get_path_of_suffix_files('./data/', '.parquet')

    save_file = './data/all_dataset.parquet'

    # 后续append写入，存在文件先删除
    if exists(save_file): 
        assert delete_file(save_file)

    cur_rows = []
    append = cur_rows.append
    
    all_cnt, keep_cnt = 0, 0
    for file in from_parquet_files:
        print('process file: {}'.format(file))

        parquet_table = pq.read_table(file)
     
        for response in progress.track(parquet_table['response'], total=parquet_table.num_rows):

            response =  response.as_py()
            all_cnt += 1

            if len(response) < min_len:
                continue

            if cut_max_len and len(response) > max_len:
                response = response[0: max_len]

            keep_cnt += 1
            append({'response': response})

            if len(cur_rows) >= groups_cnt:
                df = pd.DataFrame(cur_rows)
                write_single_parquet_file(save_file, df)
                cur_rows = []
                append = cur_rows.append
                
    # 处理末尾部分
    if len(cur_rows) > 0:
        df = pd.DataFrame(cur_rows)
        write_single_parquet_file(save_file, df)
        cur_rows = []

    log.info("merge into file: {}, 全部数据共{}行，清洗后剩余{}行".format(save_file, all_cnt, keep_cnt), save_to_file=True)
    
def remove_dataset_duplicate_rows(groups_cnt: int=50000) -> None:
    '''
    使用min_hash删除数据集中重复的部分
    '''
    from_parquet_files = '../data/563w_baidubaike/baike.parquet'

    save_file = '../data/563w_baidubaike/all_no_dulpticates.parquet'

    # 后续append写入，存在文件先删除
    if exists(save_file): 
        assert delete_file(save_file)

    cur_rows = []
    all_cnt, keep_cnt = 0, 0
    row_index = -1
    drop_dataset_duplicate = DropDatasetDuplicate(threshold=0.85, num_perm=256)
    
    parquet_table = pq.read_table(from_parquet_files)
    all_cnt = parquet_table.num_rows
    print(all_cnt)
    # 先顺序遍历获取哪些行是重复的
    for response in progress.track(parquet_table['response'], total=parquet_table.num_rows):
        row_index += 1

        doc = f"{response.as_py()}" # 将JSON格式的响应转换为Python字典
        drop_dataset_duplicate.add_doc(index=row_index, doc=doc)

    row_index = -1
    need_to_drop_indexs = drop_dataset_duplicate.get_duplicate_indexs()

    # 再顺序遍历一遍，重复的行不添加到新的数据集
    for response in progress.track(parquet_table['response'], total=parquet_table.num_rows):
        row_index += 1  # 不管有没有跳过行, row_index都必须+1

        # 重复的行跳过
        if row_index in need_to_drop_indexs:
            continue

        cur_rows.append({'response': response.as_py()})
        keep_cnt += 1

        # 分块写入
        if len(cur_rows) >= groups_cnt:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(save_file, df)
            cur_rows = []
    # 处理末尾部分，并写入
    if len(cur_rows) > 0:
        df = pd.DataFrame(cur_rows)
        write_single_parquet_file(save_file, df)
    log.info("merge into file: {}, 全部数据共{}行，文档去重后剩余{}行".format(save_file, all_cnt, keep_cnt), save_to_file=True)



def shuffle_parquet_dataset(parquet_file: str, shuffle_file: str, seed: int=23333, groups_cnt: int=65536) -> None:
    '''
    打乱一个parquet文件数据集
    '''
    if not exists(parquet_file):
        raise Exception('can not find parquet file: {}'.format(parquet_file))
    
    print('start shuffle...')
    pf =  pq.read_table(parquet_file)
    df = pf.to_pandas()
    df = df.sample(frac=1.0, replace=False, random_state=seed, axis=0)
    
    if exists(shuffle_file): 
        assert delete_file(shuffle_file)
    
    # 分块写入parquet，否则小内存读取直接OOM
    n = len(df)
    for i in range(0, n, groups_cnt):
        cur_group_df = df[i: i + groups_cnt]
        write_single_parquet_file(shuffle_file, cur_group_df)


def read_and_write_template_wiki(read_file: str, write_to_file: str, call_back: object, group_cnt: int=10000) -> None:

    log.info('process file:{}'.format(read_file), save_to_file=True)
    start = time.time()
    
    raw_line_cnt = 0
    keep_line_cnt = 0
    with progress.open(read_file, 'r', encoding='utf-8') as f_read:
        
        json_list = ujson.load(f_read)
        cur_rows = []
        append = cur_rows.append

        for line in json_list:
            try:
                #print(line)
                raw_line_cnt += 1
                write_dict = call_back(line)
                if write_dict is None: continue
                keep_line_cnt += 1
                append(write_dict)
                if len(cur_rows) >= group_cnt:
                    df = pd.DataFrame(cur_rows)
                    write_single_parquet_file(write_to_file, df)
                    cur_rows = []
                    append = cur_rows.append
            except Exception as e:
                # log.error('处理文件异常：{}, content:{}'.format(str(e), line))
                print(line)
                raise e
            # end for
            # 处理末尾部分
        if len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(write_to_file, df)
            cur_rows = []
        end = time.time()
        log.info('原始文件:{}，共{}行，处理后剩余{}行，保存到文件：{}。耗时：{:.6}s'\
                    .format(read_file, raw_line_cnt, keep_line_cnt, write_to_file, end - start), save_to_file=True)
        
def process_wiki(response_less_word: int=15) -> None:
    file_names = [
        '../data/wikipedia-cn-20230720-filtered.json',
    ]
    save_file_name = './data/wiki.parquet'
    if exists(save_file_name): 
        assert delete_file(save_file_name)

    def process_function(item: dict) -> dict:
        #print(item['completion'])
        # 数据清洗
        response = item['completion'].replace('\r','')
        response = remove_duplicate_punctuation(response)
        # 剔除短数据
        if len(response) < response_less_word:
            return None
        write_dict = {
                "response": response,
            }
        return write_dict

    for file_name in file_names:
        read_file = file_name
        read_and_write_template_wiki(read_file, save_file_name, process_function)
        

def read_and_write_template_baike(read_file: str, write_to_file: str, call_back: object, group_cnt: int=10000) -> None:

    log.info('process file:{}'.format(read_file), save_to_file=True)
    start = time.time()
    
    raw_line_cnt = 0
    keep_line_cnt = 0
    with progress.open(read_file, 'r', encoding='utf-8') as f_read:
        cur_rows = []
        append = cur_rows.append
        for line in f_read:
            try:
                #print(line)
                raw_line_cnt += 1
                write_dict = call_back(line)
                if write_dict is None: continue
                keep_line_cnt += 1
                append(write_dict)
                if len(cur_rows) >= group_cnt:
                    df = pd.DataFrame(cur_rows)
                    write_single_parquet_file(write_to_file, df)
                    cur_rows = []
                    append = cur_rows.append
            except Exception as e:
                # log.error('处理文件异常：{}, content:{}'.format(str(e), line))
                print(line)
                raise e
            # end for
            # 处理末尾部分
        if len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(write_to_file, df)
            cur_rows = []
        end = time.time()
        log.info('原始文件:{}，共{}行，处理后剩余{}行，保存到文件：{}。耗时：{:.6}s'\
                    .format(read_file, raw_line_cnt, keep_line_cnt, write_to_file, end - start), save_to_file=True)

def process_baike(response_less_word: int=15) -> None:
    file_names = [
        '../data/563w_baidubaike/563w_baidubaike.json',
    ]
    save_file_name = '../data/563w_baidubaike/baike.parquet'
    if exists(save_file_name): 
        assert delete_file(save_file_name)

    def process_function(line: str) -> dict:
        
        item = ujson.loads(line)
        item_title = item['title']
        item_sections = item ['sections']
        for data in item_sections:
            #print(item['completion'])
            # 数据清洗
            response = data['content'].replace('\r','')
            response = remove_duplicate_punctuation(response)
            # 剔除短数据
            if len(response) < response_less_word:
                return None
            response = data['title']+data['content']
            write_dict = {
                    "response": response,
                }
            return write_dict

    for file_name in file_names:
        read_file = file_name
        read_and_write_template_baike(read_file, save_file_name, process_function)

if __name__ == '__main__':

    data=open('../data/563w_baidubaike.json','r')
    for line in data.readlines()[:10]:
        print(line)
    
    process_wiki()
    process_baike()
    
    remove_dataset_duplicate_rows()

