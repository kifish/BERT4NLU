
import os,json
import random, string
from tqdm import tqdm 
from transformers import BertTokenizer
import torch 
from src.utils.utils import timer_context
from multiprocessing import Pool
import multiprocessing as mp 
import mmap


from functools import wraps
import time

from contextlib import ContextDecorator


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # start = time.clock()
        start = time.time()
        r = func(*args, **kwargs)
        # end = time.clock()
        end = time.time()
        print('[' + func.__name__ + '] used: {} s'.format(end - start)) # wall time
        return r
    return wrapper

@timing
def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    n_line = 0
    while buf.readline():
        n_line += 1
    return n_line


def save_examples_to_jsonl(examples, path):
    root_dir = os.path.dirname(path)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        
    print('saving data into {} ...'.format(path))
    with open(path, 'w', encoding = 'utf8') as f:
        for example in tqdm(examples):
            line = json.dumps(example)
            f.write(line)
            f.write('\n')
    print('done .')        
    


@timing
def read_cached_data_tqdm(file_path, debug = False, debug_num = 1000):
    examples = []
    print('reading cached data ...')
    with open(file_path, 'r', encoding='utf8') as f:
        for line in tqdm(f, total= get_num_lines(file_path)): # generator
            line = line.strip()
            example = json.loads(line)
            examples.append(example)
            del line 
            if debug and len(examples) >= debug_num:
                break
    print('num of cached examples : {}'.format(len(examples))) 
    return examples



def load_file_multiTurn(file_path, num_cands = 20, shuffle_level= 'examples', shuffle_cand = False, data_format = 'self'):
    """
    return examples, positions
    examples: list
    example: dict;
    {   'document': document,      # document:list
        'context': current_turn['context'], # current_turn['context']:list
        'response': c, # c:str
        'label': l
    } # l:int 0 or 1
    positions: [(start,end)];方便后续shuffle_level的控制
    """
    examples = []
    positions = []
    num_episode = 0
    num_example = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        document, episode = [], []
        start, end = 0, 0
        for i, line in enumerate(lines):
            episode_done = (i + 1 == len(lines) or lines[i + 1].startswith('1 ')) # 一个完整的多轮对话结束

            line = line.lstrip(string.digits).strip()
            if data_format == 'self':
                line = line.replace('your persona:', '').strip()
            else:
                line = line.replace("partner's persona:", '').strip()
            parts = line.split('\t')
            if len(parts) == 1: # 无\t分割的文本说明是document的句子
                document.append(parts[0])
            else:
                response = parts[1]
                query = parts[0]
                cands = parts[-1].split('|')
                if shuffle_cand:
                    random.shuffle(cands)
                    labels = [1 if cand == response else 0 for cand in cands]
                else:
                    labels = [1 if idx + 1 == len(cands) else 0 for idx,c in enumerate(cands)] # 优化判断
                if 'train' in file_path:
                    cands = cands[-num_cands:]
                    labels = labels[-num_cands:]

                num_example += 1
                current_turn = {'context': [query], 'response': response} # 当前对话
                if episode:
                    current_turn['context'] = episode[-1]['context'] + [episode[-1]['response']] + current_turn['context']
                episode.append(current_turn)
                for c, l in zip(cands, labels):
                    examples.append({'document': document,      # document:list
                                     'context': current_turn['context'], # current_turn['context']: list
                                     'response': c, # c:str
                                     'label': l}) # l:int 0 or 1
                    end += 1
                # examples level shuffle !!!
                if shuffle_level == 'examples':
                    positions.append((start, end))
                    start = end

            if episode_done:
                num_episode += 1
                document, episode = [], []
                # session level shuffle !!!
                if shuffle_level == 'session': # 一个完整的多轮对话结束
                    positions.append((start, end))
                    start = end
    
    print('num_episode(session) : {}'.format(num_episode))
    print('num_example : {}'.format(num_example)) # 这里的example没考虑num_cands; 最终的examples cnt是num_example*num_cands
    return examples, positions

# knowledge context response
# segment embedding
# 目前实现的版本tokenization速度太慢, 会block训练和eval
class Batcher(object):
    
    def __init__(self, config: dict) -> None:        
        self.bert_tokenizer = BertTokenizer.from_pretrained(config['name'], cache_dir = config['cache_dir'])
        self.logger = config['logger']
        self.block_size = config['block_size']
        self.seq_len = self.block_size - 2
        
        SPECIAL_TOKENS_DICT = {'additional_special_tokens': ["<user1>", "<user2>", "<knowledge>"]}
        self.bert_tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        
        self.pad_id = self.bert_tokenizer.pad_token_id  # 0
        self.user_ids = [self.bert_tokenizer.convert_tokens_to_ids('<user1>'), self.bert_tokenizer.convert_tokens_to_ids('<user2>')]
        self.know_id = self.bert_tokenizer.convert_tokens_to_ids('<knowledge>') # knowledge: persona
        self.cls_id = self.bert_tokenizer.convert_tokens_to_ids('[CLS]') # 101
        self.sep_id = self.bert_tokenizer.convert_tokens_to_ids('[SEP]') # 102
        
        print('user_ids: {}'.format(self.user_ids))
        print('know_id: {}'.format(self.know_id))
        print('bos: {} {}, eos: {} {}, unk: {} {}'.format(self.bert_tokenizer.bos_token, self.bert_tokenizer.bos_token_id, \
                                                        self.bert_tokenizer.eos_token, self.bert_tokenizer.eos_token_id, \
                                                        self.bert_tokenizer.unk_token, self.bert_tokenizer.unk_token_id)) 

        self.logger.info('vocab size : {}'.format(len(self.bert_tokenizer))) 

        self.device = torch.device('cuda' if config['cuda'] else 'cpu')
        
    def tokenize(self, text):
        return self.bert_tokenizer.encode(text, text_pair=None, add_special_tokens=False)
    
        
    def __call__(self, b_examples):
        b_input_ids = [] # 2-d list 
        b_type_ids = [] # 2-d list
        b_label = [] # 1-d list

        for ex in b_examples: # 遍历单条样本            
            k_list = ex['document']  # 1-d list
            knowledge_ids = [self.tokenize(k) for k in k_list] # 2-d list
            flatten_knowledge_ids = []
            for i, k in enumerate(knowledge_ids):
                if i != 0:
                    flatten_knowledge_ids.append(self.know_id)
                
                flatten_knowledge_ids.extend(k)
            
            knowledge_type_ids = len(flatten_knowledge_ids) * [0]

            context_list = ex['context']
            context_ids = [self.tokenize(c) for c in context_list] # 2-d list

            flatten_context_ids = []
            context_type_ids = []
            for idx, utterance_ids in enumerate(context_ids):
                idx = idx % 2
                tmp = [self.user_ids[idx]] + utterance_ids
                flatten_context_ids += tmp
                context_type_ids += len(tmp) * [0]
                
            resp = ex['response']
            resp = self.tokenize(resp) 
            response_ids = [self.user_ids[1]] + resp
            response_type_ids = len(response_ids) * [1]
            
            input_ids = flatten_knowledge_ids + flatten_context_ids + response_ids
            # input_ids = input_ids[-self.block_size:] # 这里截断处理比较粗糙
            
            input_ids = input_ids[-self.seq_len:]
            input_ids = [self.cls_id] + input_ids + [self.sep_id]
            input_ids = input_ids + [self.pad_id] * (self.block_size - len(input_ids)) 

            input_type_ids = knowledge_type_ids + context_type_ids + response_type_ids
            input_type_ids = input_type_ids[-self.seq_len:]
            input_type_ids = [0] + input_type_ids + [0] # cls sep 对应的type暂时设定为0
            input_type_ids = input_type_ids + [0] * (self.block_size - len(input_type_ids))  
            
            b_input_ids.append(input_ids)
            b_type_ids.append(input_type_ids)
            b_label.append(ex['label'])
            
        b_input_ids = torch.tensor(b_input_ids, device = self.device, dtype=torch.long) # 2-d tensor 
        b_masks = torch.ne(b_input_ids, 0).float() # 2-d tensor 
        b_type_ids = torch.tensor(b_type_ids, device = self.device, dtype=torch.long) # 2-d tensor 
        b_label = torch.tensor(b_label, device=self.device, dtype=torch.long) # 1-d tensor 
        return b_input_ids, b_masks, b_type_ids, b_label
    

# Batcher的cache 版本
class Batcher_v2(object):
    
    def __init__(self, config: dict) -> None:        
        self.device = torch.device('cuda' if config['cuda'] else 'cpu')
        
    def __call__(self, b_examples):
        b_input_ids = [] # 2-d list 
        b_type_ids = [] # 2-d list
        b_label = [] # 1-d list

        for ex in b_examples: # 遍历单条样本  
            input_ids = ex['input_ids']
            input_type_ids = ex['input_type_ids']   
            b_input_ids.append(input_ids)
            b_type_ids.append(input_type_ids)
            b_label.append(ex['label'])
            
        b_input_ids = torch.tensor(b_input_ids, device = self.device, dtype=torch.long) # 2-d tensor 
        b_masks = torch.ne(b_input_ids, 0).float() # 2-d tensor 
        b_type_ids = torch.tensor(b_type_ids, device = self.device, dtype=torch.long) # 2-d tensor 
        b_label = torch.tensor(b_label, device=self.device, dtype=torch.long) # 1-d tensor 
        return b_input_ids, b_masks, b_type_ids, b_label
    



class Processor(object):
    def __init__(self, config: dict) -> None:
        self.bert_tokenizer = BertTokenizer.from_pretrained(config['name'], cache_dir = config['cache_dir'])
        self.block_size = config['block_size']
        self.seq_len = self.block_size - 2
        
        SPECIAL_TOKENS_DICT = {'additional_special_tokens': ["<user1>", "<user2>", "<knowledge>"]}
        self.bert_tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        
        self.pad_id = self.bert_tokenizer.pad_token_id  # 0
        self.user_ids = [self.bert_tokenizer.convert_tokens_to_ids('<user1>'), self.bert_tokenizer.convert_tokens_to_ids('<user2>')]
        self.know_id = self.bert_tokenizer.convert_tokens_to_ids('<knowledge>') # knowledge: persona
        self.cls_id = self.bert_tokenizer.convert_tokens_to_ids('[CLS]') # 101
        self.sep_id = self.bert_tokenizer.convert_tokens_to_ids('[SEP]') # 102
        
        
    def tokenize(self, text):
        return self.bert_tokenizer.encode(text, text_pair=None, add_special_tokens=False)
    
    def __call__(self, ex):
        k_list = ex['document']  # 1-d list
        knowledge_ids = [self.tokenize(k) for k in k_list] # 2-d list
        flatten_knowledge_ids = []
        for i, k in enumerate(knowledge_ids):
            if i != 0:
                flatten_knowledge_ids.append(self.know_id)
            
            flatten_knowledge_ids.extend(k)
        
        knowledge_type_ids = len(flatten_knowledge_ids) * [0]

        context_list = ex['context']
        context_ids = [self.tokenize(c) for c in context_list] # 2-d list

        flatten_context_ids = []
        context_type_ids = []
        for idx, utterance_ids in enumerate(context_ids):
            idx = idx % 2
            tmp = [self.user_ids[idx]] + utterance_ids
            flatten_context_ids += tmp
            context_type_ids += len(tmp) * [0]
            
        resp = ex['response']
        resp = self.tokenize(resp) 
        response_ids = [self.user_ids[1]] + resp
        response_type_ids = len(response_ids) * [1]
        
        input_ids = flatten_knowledge_ids + flatten_context_ids + response_ids
        # input_ids = input_ids[-self.block_size:] # 这里截断处理比较粗糙
        
        input_ids = input_ids[-self.seq_len:]
        input_ids = [self.cls_id] + input_ids + [self.sep_id]
        input_ids = input_ids + [self.pad_id] * (self.block_size - len(input_ids)) 

        input_type_ids = knowledge_type_ids + context_type_ids + response_type_ids
        input_type_ids = input_type_ids[-self.seq_len:]
        input_type_ids = [0] + input_type_ids + [0] # cls sep 对应的type暂时设定为0
        input_type_ids = input_type_ids + [0] * (self.block_size - len(input_type_ids))  
        
        ret = {
            'input_ids':input_ids,
            'input_type_ids':input_type_ids,
            'label': ex['label']
        }

        return ret 
    
        
        
        
def cache_data(src_path, tgt_path):
    examples, positions = load_file_multiTurn(src_path, num_cands = 20, shuffle_level= 'examples', shuffle_cand = False, data_format = 'self')
    config = {  # 处理数据
        'name': 'bert-base-uncased',
        'cache_dir': 'resource/pretrained_models/bert',
        'block_size': 256,
    }
    
    process_fn = Processor(config)
    process_num = mp.cpu_count()
    chunksize = int(len (examples) / process_num) + 1
    with timer_context('convert examples to ids'):
        with Pool() as p:
            processed_examples = p.map(process_fn, examples, chunksize = chunksize)        
        print('done.')

    save_examples_to_jsonl(processed_examples, tgt_path)
    
    


# knowledge context response
# 加入特殊的token type embedding
class Batcher_v3(object):
    
    def __init__(self, config: dict) -> None:        
        self.bert_tokenizer = BertTokenizer.from_pretrained(config['name'], cache_dir = config['cache_dir'])
        self.logger = config['logger']
        self.block_size = config['block_size']
        self.seq_len = self.block_size - 2
        
        SPECIAL_TOKENS_DICT = {'additional_special_tokens': ["<user1>", "<user2>", "<knowledge>"]}
        self.bert_tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        
        self.pad_id = self.bert_tokenizer.pad_token_id  # 0
        self.user_ids = [self.bert_tokenizer.convert_tokens_to_ids('<user1>'), self.bert_tokenizer.convert_tokens_to_ids('<user2>')]
        self.know_id = self.bert_tokenizer.convert_tokens_to_ids('<knowledge>') # knowledge: persona
        self.cls_id = self.bert_tokenizer.convert_tokens_to_ids('[CLS]') # 101
        self.sep_id = self.bert_tokenizer.convert_tokens_to_ids('[SEP]') # 102
        
        print('user_ids: {}'.format(self.user_ids))
        print('know_id: {}'.format(self.know_id))
        print('bos: {} {}, eos: {} {}, unk: {} {}'.format(self.bert_tokenizer.bos_token, self.bert_tokenizer.bos_token_id, \
                                                        self.bert_tokenizer.eos_token, self.bert_tokenizer.eos_token_id, \
                                                        self.bert_tokenizer.unk_token, self.bert_tokenizer.unk_token_id)) 

        self.logger.info('vocab size : {}'.format(len(self.bert_tokenizer))) 

        self.device = torch.device('cuda' if config['cuda'] else 'cpu')
        
    def tokenize(self, text):
        return self.bert_tokenizer.encode(text, text_pair=None, add_special_tokens=False)
    
        
    def __call__(self, b_examples):
        b_input_ids = [] # 2-d list 
        b_type_ids = [] # 2-d list
        b_label = [] # 1-d list

        for ex in b_examples: # 遍历单条样本            
            k_list = ex['document']  # 1-d list
            knowledge_ids = [self.tokenize(k) for k in k_list] # 2-d list
            flatten_knowledge_ids = []
            for i, k in enumerate(knowledge_ids):
                if i != 0:
                    flatten_knowledge_ids.append(self.know_id)
                
                flatten_knowledge_ids.extend(k)
            
            knowledge_type_ids = len(flatten_knowledge_ids) * [self.know_id]

            context_list = ex['context']
            context_ids = [self.tokenize(c) for c in context_list] # 2-d list

            flatten_context_ids = []
            context_type_ids = []
            for idx, utterance_ids in enumerate(context_ids):
                idx = idx % 2
                tmp = [self.user_ids[idx]] + utterance_ids
                flatten_context_ids += tmp
                context_type_ids += len(tmp) * [self.user_ids[idx]] # 与Batcher 不一样
                
            resp = ex['response']
            resp = self.tokenize(resp) 
            response_ids = [self.user_ids[1]] + resp
            response_type_ids = len(response_ids) * [self.user_ids[1]]
            
            input_ids = flatten_knowledge_ids + flatten_context_ids + response_ids
            # input_ids = input_ids[-self.block_size:] # 这里截断处理比较粗糙
            input_ids = input_ids[-self.seq_len:]
            input_ids = [self.cls_id] + input_ids + [self.sep_id]
            input_ids = input_ids + [self.pad_id] * (self.block_size - len(input_ids)) 

            input_type_ids = knowledge_type_ids + context_type_ids + response_type_ids
            input_type_ids = input_type_ids[-self.seq_len:]
            input_type_ids = [0] + input_type_ids + [0] # cls sep 对应的type暂时设定为0
            input_type_ids = input_type_ids + [0] * (self.block_size - len(input_type_ids))  
            
            b_input_ids.append(input_ids)
            b_type_ids.append(input_type_ids)
            b_label.append(ex['label'])
            
        b_input_ids = torch.tensor(b_input_ids, device = self.device, dtype=torch.long) # 2-d tensor 
        b_masks = torch.ne(b_input_ids, 0).float() # 2-d tensor 
        b_type_ids = torch.tensor(b_type_ids, device = self.device, dtype=torch.long) # 2-d tensor 
        b_label = torch.tensor(b_label, device=self.device, dtype=torch.long) # 1-d tensor 
        return b_input_ids, b_masks, b_type_ids, b_label
    



class Stat():
    
    
    def __init__(self, config: dict) -> None:        
        self.bert_tokenizer = BertTokenizer.from_pretrained(config['name'], cache_dir = config['cache_dir'])
        self.block_size = config['block_size']
        self.seq_len = self.block_size - 2
        self.path = config['path']

        SPECIAL_TOKENS_DICT = {'additional_special_tokens': ["<user1>", "<user2>", "<knowledge>"]}
        self.bert_tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        
        self.pad_id = self.bert_tokenizer.pad_token_id  # 0
        self.user_ids = [self.bert_tokenizer.convert_tokens_to_ids('<user1>'), self.bert_tokenizer.convert_tokens_to_ids('<user2>')]
        self.know_id = self.bert_tokenizer.convert_tokens_to_ids('<knowledge>') # knowledge: persona
        self.cls_id = self.bert_tokenizer.convert_tokens_to_ids('[CLS]') # 101
        self.sep_id = self.bert_tokenizer.convert_tokens_to_ids('[SEP]') # 102
        
        print('user_ids: {}'.format(self.user_ids))
        print('know_id: {}'.format(self.know_id))
        print('bos: {} {}, eos: {} {}, unk: {} {}'.format(self.bert_tokenizer.bos_token, self.bert_tokenizer.bos_token_id, \
                                                        self.bert_tokenizer.eos_token, self.bert_tokenizer.eos_token_id, \
                                                        self.bert_tokenizer.unk_token, self.bert_tokenizer.unk_token_id)) 

        print('vocab size : {}'.format(len(self.bert_tokenizer))) # 30522 + 3 = 30525

                
    def tokenize(self, text):
        return self.bert_tokenizer.encode(text, text_pair=None, add_special_tokens=False)
    
        
    def process(self, examples):
        examples_cnt = len(examples)
        truncated_cnt = 0
        block_lens = []
        for ex in tqdm(examples): # 遍历单条样本
   
            k_list = ex['document']  # 1-d list
            knowledge_ids = [self.tokenize(k) for k in k_list] # 2-d list
            flatten_knowledge_ids = []
            for i, k in enumerate(knowledge_ids):
                if i != 0:
                    flatten_knowledge_ids.append(self.know_id)
                
                flatten_knowledge_ids.extend(k)
            
            knowledge_type_ids = len(flatten_knowledge_ids) * [self.know_id]

            context_list = ex['context']
            context_ids = [self.tokenize(c) for c in context_list] # 2-d list

            flatten_context_ids = []
            context_type_ids = []
            for idx, utterance_ids in enumerate(context_ids):
                idx = idx % 2
                tmp = [self.user_ids[idx]] + utterance_ids
                flatten_context_ids += tmp
                context_type_ids += len(tmp) * [self.user_ids[idx]]
                
            resp = ex['response']
            resp = self.tokenize(resp) 
            response_ids = [self.user_ids[1]] + resp
            response_type_ids = len(response_ids) * [self.user_ids[1]]
            
            input_ids = flatten_knowledge_ids + flatten_context_ids + response_ids
            
            if len(input_ids) > self.block_size:
                truncated_cnt += 1
            
            block_lens.append(len(input_ids))



        avg_block_len = sum(block_lens) / len(block_lens)
        
        print('examples_cnt : {}'.format(examples_cnt))
        print('truncated_cnt : {}'.format(truncated_cnt))
        print('truncated_ratio : {}'.format(truncated_cnt / examples_cnt))
        print('avg_block_len : {}'.format(avg_block_len))



    def process_parallel(self, ex):
        k_list = ex['document']  # 1-d list
        knowledge_ids = [self.tokenize(k) for k in k_list] # 2-d list
        flatten_knowledge_ids = []
        for i, k in enumerate(knowledge_ids):
            if i != 0:
                flatten_knowledge_ids.append(self.know_id)
            
            flatten_knowledge_ids.extend(k)
        
        knowledge_type_ids = len(flatten_knowledge_ids) * [self.know_id]

        context_list = ex['context']
        context_ids = [self.tokenize(c) for c in context_list] # 2-d list

        flatten_context_ids = []
        context_type_ids = []
        for idx, utterance_ids in enumerate(context_ids):
            idx = idx % 2
            tmp = [self.user_ids[idx]] + utterance_ids
            flatten_context_ids += tmp
            context_type_ids += len(tmp) * [self.user_ids[idx]]
            
        resp = ex['response']
        resp = self.tokenize(resp) 
        response_ids = [self.user_ids[1]] + resp
        response_type_ids = len(response_ids) * [self.user_ids[1]]
        input_ids = flatten_knowledge_ids + flatten_context_ids + response_ids
        return len(input_ids)
    

    def stat(self):
        examples, _ = load_file_multiTurn(self.path, num_cands = 20, shuffle_level= 'examples', shuffle_cand = False, data_format = 'self')
        self.process(examples)
        
        

    def stat_fast(self):
        examples, _ = load_file_multiTurn(self.path, num_cands = 20, shuffle_level= 'examples', shuffle_cand = False, data_format = 'self')
        process_num = mp.cpu_count()
        chunksize = int(len (examples) / process_num) + 1
        with timer_context('convert tokens to ids'):
            with Pool() as p:
                block_lens = p.map(self.process_parallel, examples, chunksize = chunksize)        
            print('done.')
        
        avg_block_len = sum(block_lens) / len(block_lens)
        examples_cnt = len(examples)
        truncated_cnt = 0
        for block_len in block_lens:
            if block_len > self.block_size:
                truncated_cnt += 1
                
        print('examples_cnt : {}'.format(examples_cnt))
        print('truncated_cnt : {}'.format(truncated_cnt))
        print('truncated_ratio : {}'.format(truncated_cnt / examples_cnt))
        print('avg_block_len : {}'.format(avg_block_len))


 
if __name__ == '__main__':
    # stat 
    # stat_config = {
    #     'path':'resource/data/personachat/train_self_original.txt',
    #     'name': 'bert-base-uncased',
    #     'cache_dir': 'resource/models/bert',
    #     'block_size': 128,
    # }
    
    # stat = Stat(stat_config)
    # # stat.stat() # 1314380个examples, 单进程大概需要1个小时
    # stat.stat_fast()

    # user_ids: [30522, 30523]
    # know_id: 30524
    # Using bos_token, but it is not set yet.
    # Using eos_token, but it is not set yet.
    # bos: None None, eos: None None, unk: [UNK] 100
    # vocab size : 30525
    # num_episode(session) : 8939
    # num_example : 65719
    # convert tokens to ids ...
    # done.
    # Processing time for [convert tokens to ids] is: 215.29896306991577 seconds
    # examples_cnt : 1314380
    # truncated_cnt : 759396
    # truncated_ratio : 0.5777598563581309
    # avg_block_len : 146.870456032502
    # 大概加速18倍
    
    # -----------------------------------
    
    # stat_config = {
    #     'path':'resource/data/personachat/train_self_original.txt',
    #     'name': 'bert-base-uncased',
    #     'cache_dir': 'resource/models/bert',
    #     'block_size': 160,
    # }
    
    # stat = Stat(stat_config)
    # stat.stat_fast()
    
    # user_ids: [30522, 30523]
    # know_id: 30524
    # Using bos_token, but it is not set yet.
    # Using eos_token, but it is not set yet.
    # bos: None None, eos: None None, unk: [UNK] 100
    # vocab size : 30525
    # num_episode(session) : 8939
    # num_example : 65719
    # convert tokens to ids ...
    # done.
    # Processing time for [convert tokens to ids] is: 218.4616334438324 seconds
    # examples_cnt : 1314380
    # truncated_cnt : 545082
    # truncated_ratio : 0.4147065536602809
    # avg_block_len : 146.870456032502
    
    
    # -----------------------------------
    
    # stat_config = {
    #     'path':'resource/data/personachat/train_self_original.txt',
    #     'name': 'bert-base-uncased',
    #     'cache_dir': 'resource/models/bert',
    #     'block_size': 180,
    # }
    
    # stat = Stat(stat_config)
    # stat.stat_fast()
    
    # user_ids: [30522, 30523]
    # know_id: 30524
    # Using bos_token, but it is not set yet.
    # Using eos_token, but it is not set yet.
    # bos: None None, eos: None None, unk: [UNK] 100
    # vocab size : 30525
    # num_episode(session) : 8939
    # num_example : 65719
    # convert tokens to ids ...
    # done.
    # Processing time for [convert tokens to ids] is: 227.76235818862915 seconds
    # examples_cnt : 1314380
    # truncated_cnt : 411832
    # truncated_ratio : 0.31332795690743925
    # avg_block_len : 146.870456032502
    
    # -----------------------------------
    # stat_config = {
    #     'path':'resource/data/personachat/train_self_original.txt',
    #     'name': 'bert-base-uncased',
    #     'cache_dir': 'resource/models/bert',
    #     'block_size': 256,
    # }
    
    # stat = Stat(stat_config)
    # stat.stat_fast()
    
    # user_ids: [30522, 30523]
    # know_id: 30524
    # Using bos_token, but it is not set yet.
    # Using eos_token, but it is not set yet.
    # bos: None None, eos: None None, unk: [UNK] 100
    # vocab size : 30525
    # num_episode(session) : 8939
    # num_example : 65719
    # convert tokens to ids ...
    # done.
    # Processing time for [convert tokens to ids] is: 220.68003034591675 seconds
    # examples_cnt : 1314380
    # truncated_cnt : 52833
    # truncated_ratio : 0.04019613810313608
    # avg_block_len : 146.870456032502
    # -----------------------------------

    cache_data('resource/data/personachat/train_self_original.txt', 'resource/data/personachat/processed/processed_train_self_original.jsonl')
    
    cache_data('resource/data/personachat/valid_self_original.txt', 'resource/data/personachat/processed/processed_valid_self_original.jsonl')
    
    cache_data('resource/data/personachat/test_self_original.txt', 'resource/data/personachat/processed/processed_test_self_original.jsonl')
    
    
    
        