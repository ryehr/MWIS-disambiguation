import torch
import torch.nn.functional as F
import os
import time
import argparse
from tqdm import trange
from transformers import GPT2LMHeadModel, GPT2Config, CpmTokenizer
from utils import top_k_top_p_filtering, set_logger
from os.path import join, exists
import random
import math
import csv
import codecs
from decimal import *
import numpy as np
from collections import deque

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class node(object):
    def __init__(self,index,word,weight,neighbour):
        self.index=index
        self.word=word
        self.weight=weight
        self.neighbour=neighbour

getcontext().prec = 400

def find_connected_components(Nodes):
    visited = set()
    components = []
    for i in range(len(Nodes)):
        if i not in visited:
            component = []
            dfs(Nodes, i, visited, component)
            components.append(component)
    return components

def dfs(Nodes, i, visited, component):
    visited.add(i)
    component.append(i)
    for neighbor in Nodes[i].neighbour:
        if neighbor not in visited:
            dfs(Nodes, neighbor, visited, component)

def BFS_Forest(Forest_nodes,Nodes):
    start=Forest_nodes[0]

    queue = deque()
    visited = set()
    queue.append(start)
    queue.append(None)
    visited.add(start)
    result = []
    level = []
    while queue:
        node = queue.popleft()
        if node!=None:
            level.append(node)
            for neighbor in Nodes[node].neighbour:
                if neighbor not in Forest_nodes:
                    continue
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)
            if  queue[0]==None:
                queue.popleft()
                result.append(level)
                level = []
                if queue:
                    queue.append(None)
        else:
            if queue:
                queue.append(None)
    return result,visited

def find_MWIS(Nodes):

    visited=[]
    Forest_nodes=[]
    for i in range(len(Nodes)):
        visited.append(i)
        index=0
        for nei in Nodes[i].neighbour:
            if nei in visited:
                index+=1
        if index<=1:
            Forest_nodes.append(i)

    # print(Forest_nodes)
    Layer=[]

    while Forest_nodes!=[]:

        layer_result,visited=BFS_Forest(Forest_nodes,Nodes)
        # print(visited)
        for node in visited:
            Forest_nodes.remove(node)
        # print(Forest_nodes)

        Layer.append(layer_result)
        # print(Layer)
    return Layer

def DP(subtree,Nodes):
    if len(subtree)==1:
        return subtree[0]

    sum_weight=[]
    sub_final=[]
    group_final=[]
    # print('Original')
    for item in subtree:
        temp=0
        for i in item:
            temp+=Nodes[i].weight
            # print(Nodes[i].word,Nodes[i].weight)
        sum_weight.append(temp)
    N=len(sum_weight)
    dp=[0]*(N+1)
    dp[0]=0
    dp[1]=sum_weight[0]
    # print(sum_weight)

    for k in range(2,N+1):
        # print(k)
        dp[k]=max(dp[k-1],sum_weight[k-1]+dp[k-2])

    k=N
    while k>=1:
        if dp[k]==dp[k-1]:
            k-=1
        else:
            group_final.append(k-1)
            k-=2
    # print('Final:')
    for i in group_final:
        for j in subtree[i]:
            sub_final.append(j)
            # print(Nodes[j].word, Nodes[j].weight)
    return sub_final


def MWIS(chars_freqs,valid_words):
    if args.approach =='greedy':
        Nodes=[]
        MWIS_Nodes=[]
        for i in range(len(valid_words)):
            neighbour=[]
            for j in range(len(valid_words)):
                if i!=j and (valid_words[i].startswith(valid_words[j]) or valid_words[j].startswith(valid_words[i])):
                    neighbour.append(j)
            Nodes.append(node(i,valid_words[i],chars_freqs[i][1],neighbour))
        # for n in Nodes:
        #     if n.neighbour!=[]:
        #         print(n.word+'->',end='')
        #         for nei in n.neighbour:
        #             print(valid_words[nei]+'/',end='')
        #         print()

        # components=find_connected_components(Nodes)
        #
        # final_components=[]
        # for component in components:
        #     if len(component)==1:
        #         final_components.append(component[0])
        #     if len(component)==2:
        #         if chars_freqs[component[0]][1]>chars_freqs[component[1]][1]:
        #             final_components.append(component[0])
        #         else:
        #             final_components.append(component[1])
        #     if len(component)>2:
        Layer=find_MWIS(Nodes)
        # print(Layer)
        final_index=[]
        for subtree in Layer:
            sub_final=DP(subtree,Nodes)
            final_index.extend(sub_final)
            # print(final_index)
        new_chars_freqs=[]
        new_valid_words=[]
        for i in range(len(chars_freqs)):
            if i in final_index:
                new_chars_freqs.append(chars_freqs[i])
                new_valid_words.append(valid_words[i])
        return new_chars_freqs,new_valid_words
    else:
        Nodes = []
        for i in range(len(valid_words)):
            neighbour = []
            for j in range(len(valid_words)):
                if i != j and (valid_words[i].startswith(valid_words[j]) or valid_words[j].startswith(valid_words[i])):
                    neighbour.append(j)
            Nodes.append(node(i, valid_words[i], chars_freqs[i][1], neighbour))
        # for n in Nodes:
        #     if n.neighbour!=[]:
        #         print(n.word+'->',end='')
        #         for nei in n.neighbour:
        #             print(valid_words[nei]+'/',end='')
        #         print()

        components = find_connected_components(Nodes)

        final_index = []
        for component in components:
            # print(component)
            component_sum = 0
            final_component_indicator = dict()
            component_indicator = dict()
            for num in range(2 ** len(component)):
                binary = bin(num)[2:].zfill(len(component))
                # print('Binary:',binary)
                for i in range(len(component)):
                    # print(i)
                    component_indicator[component[i]] = int(binary[i])
                flag = 0
                for n in component:
                    if component_indicator[n] == 1:
                        for other in component:
                            if other in Nodes[n].neighbour and component_indicator[other] == 1:
                                flag = 1
                                break
                    if flag == 1:
                        break
                if flag == 0:
                    sum = 0
                    for n in component:
                        sum += Nodes[n].weight * component_indicator[n]
                    if sum > component_sum:
                        component_sum = sum
                        final_component_indicator = component_indicator
            for i in component:
                if final_component_indicator[i] == 1:
                    final_index.append(i)

        # print(Layer)


        new_chars_freqs = []
        new_valid_words = []
        for i in range(len(chars_freqs)):
            if i in final_index:
                new_chars_freqs.append(chars_freqs[i])
                new_valid_words.append(valid_words[i])
        return new_chars_freqs, new_valid_words


def decode_next_token(input_ids,current_max,current_num,ste_text):

    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    # next_token_logits表示最后一个token的hidden_state对应的prediction_scores,也就是模型要预测的下一个token的概率
    next_token_logits = logits[0, -1, :]
    next_token_logits = next_token_logits / args.temperature

    next_token_logits[unk_id] = -float('Inf')
    #filtered_logits=next_token_logits
    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)

    flag=0



    if args.num_samples==0:
        word_prob = F.softmax(filtered_logits, dim=-1)
        setup_seed(args.seed)
        next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=args.num_samples)
        setup_seed(args.seed)
        add_prob = math.log(word_prob[next_token_id[0]])
    else:
        sum_prob = Decimal(0.0)
        word_prob=F.softmax(filtered_logits, dim=-1)


        next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=args.num_samples)
        chars_freqs=[]
        valid_words=[]
        for id in next_token_id:
            if id.item()==8:
                continue
            elif tokenizer.convert_ids_to_tokens(id.item())[0]=='▁':
                if tokenizer.convert_ids_to_tokens(id.item())[1:] in valid_words:
                    continue
            else:
                if '▁'+tokenizer.convert_ids_to_tokens(id.item()) in valid_words:
                    continue
            chars_freqs.append([id,float(word_prob[id].data)])
            valid_words.append(tokenizer.convert_ids_to_tokens(id.item()))

        for i in range(len(valid_words)):
            if valid_words[i][0]=='▁':
                temp=valid_words[i][1:]
                valid_words[i]=temp
        # for i in range(len(chars_freqs)):
        #     print('[',valid_words[i],chars_freqs[i][1],']',end=',')
        # print()

        # for i in range(len(chars_freqs)):
        #     print('[',valid_words[i],chars_freqs[i][1],']',end=',')
        # print('____________________________')
        # new_prob={}
        # # print(valid_words)
        # for id in chars_freqs:
        #     new_prob[id[0].item()]=id[1]
        #     sum_prob+=Decimal(id[1])
        #
        #
        #
        # init_current=Decimal(current_num)
        # init_max=Decimal(current_max)
        # current_num=Decimal(current_num)
        # index=0
        # ambiguity = 0
        # for id in  chars_freqs:
        #     #print((new_prob[id.item()]))
        #     if ste_text.startswith(valid_words[index]):
        #         for j in range(len(valid_words)):
        #             if (j!=index) and (valid_words[j].startswith(valid_words[index]) or valid_words[index].startswith(valid_words[j])):
        #                 ambiguity=1
        #                 break
        #         if ambiguity==0:
        #             next_token_id = torch.tensor([id[0]])
        #             current_max=(Decimal(id[1])/Decimal(sum_prob))*Decimal(current_max)
        #             # print('No ambiguity')
        #             # print(sum_prob)
        #             #print('[', current_num, ',', Decimal(current_num) + current_max,']')
        #             print(valid_words)
        #             return next_token_id,current_num,current_max,ste_text[len(valid_words[index]):]
        #         else:
        #             # print('Ambiguity')
        #             break
        #     else:
        #         current_num += (Decimal(id[1])/Decimal(sum_prob))*Decimal(current_max)
        #     index+=1
        # # ambiguity=1
        index=0
        chars_freqs,valid_words=MWIS(chars_freqs,valid_words)
        new_prob={}
        sum_prob = Decimal(0.0)
        for id in chars_freqs:
            new_prob[id[0].item()]=id[1]
            sum_prob+=Decimal(id[1])

        for id in  chars_freqs:
            #print((new_prob[id.item()]))

            if ste_text.startswith(valid_words[index]):
                next_token_id = torch.tensor([id[0]])
                current_max=(Decimal(id[1])/Decimal(sum_prob))*Decimal(current_max)
                # print(sum_prob)
                #print(current_max)
                #print('[', current_num, ',', Decimal(current_num) + current_max,']')
                # print(valid_words)
                return next_token_id,current_num,current_max,ste_text[len(valid_words[index]):]
            else:
                current_num += (Decimal(id[1])/Decimal(sum_prob))*Decimal(current_max)
            index += 1
        print('没找到，错误！')








def decode(ste_text):
    title_ids = tokenizer.encode(title, add_special_tokens=False)
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    input_ids = title_ids + [sep_id] + context_ids
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    cur_len = len(input_ids)



    current_num = Decimal(0)
    current_max = Decimal(2 ** args.message_length)

    while True:

        next_token_id,current_num, current_max, ste_text = decode_next_token(input_ids[:, -args.context_len:],Decimal(current_max),Decimal(current_num),ste_text)
        # print(ste_text)
        if ste_text=='':
            break

        input_ids = torch.cat((input_ids, next_token_id.unsqueeze(0)), dim=1)


    # print(len(word_list),drop_pad,drop,input_len)
    return bin(math.ceil(Decimal(current_num)))[2:].zfill(args.message_length)



if __name__ == '__main__':
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False)
    parser.add_argument('--temperature', default=1.0, type=float, required=False)
    parser.add_argument('--topk', default=64, type=int, required=False)
    parser.add_argument('--topp', default=0.0, type=float, required=False)
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False)
    parser.add_argument('--context_len', default=200, type=int, required=False)
    parser.add_argument('--max_len', default=200, type=int, required=False)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--model_path', type=str, default='model/zuowen_epoch40')
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--context', type=str, default='', help='prompt')
    parser.add_argument('--num_samples', type=int, default=64, help='candidate size')
    parser.add_argument('--secret', type=str, default='bit_stream.txt')
    parser.add_argument('--generate_num', type=int, default=1)
    parser.add_argument('--approach', type=str, default='greedy', help='greedy or enumerate')
    parser.add_argument('--seed', type=int, default=18, help='随机种子')
    parser.add_argument('--ste', type=str, default='stego.txt', help='隐写文本文件')
    parser.add_argument('--message_length', type=int, default=256, help='message长度')

    args = parser.parse_args()
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda:0' if args.cuda else 'cpu'
    # device = 'cpu'
    with open(args.ste, "r") as f:  # 打开文件
        ste_text = f.read()

    tokenizer = CpmTokenizer(vocab_file="vocab/chinese_vocab.model")
    eod_id = tokenizer.convert_tokens_to_ids("<eod>")
    sep_id = tokenizer.sep_token_id
    unk_id = tokenizer.unk_token_id


    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.eval()
    model = model.to(device)

    title = args.title
    context = args.context



    for i in range(args.generate_num):
        StartTime = time.time()
        message= decode(ste_text)
        EndTime = time.time()
        print('Embedding time={} s'.format(format(EndTime-StartTime,'.4f')))
        print('Extracted message:{}'.format(message))


