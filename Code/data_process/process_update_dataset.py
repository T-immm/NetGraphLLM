# 2000张图 每张图5个样本
# 总共1000个

# label ---> cur_weight
# node_id config_desc task query labels=cur_weight 

import os
import json
import glob
import random
import pickle
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
from datasets import Dataset, DatasetDict


def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def find_differences(pre_matrix, cur_matrix):
    pre_matrix = np.array(pre_matrix)
    cur_matrix = np.array(cur_matrix)
    
    differences = []

    # 找出不同元素的位置
    for row in range(pre_matrix.shape[0]):
        for col in range(pre_matrix.shape[1]):
            if pre_matrix[row, col] != cur_matrix[row, col]:
                differences.append([int(row), int(col), int(cur_matrix[row, col])])
    
    return differences


def convert_delay(delay):
    if 'ms' in delay:
        return int(delay.replace('ms', ''))
    elif 's' in delay:
        return int(delay.replace('s', '')) * 1000
    else:
        return int(delay)


def convert_datarate(datarate):
    if 'Gbps' in datarate:
        return float(datarate.replace('Gbps', '')) * 1000
    elif 'Mbps' in datarate:
        return float(datarate.replace('Mbps', '')) * 100
    elif 'kbps' in datarate:
        return float(datarate.replace('kbps', ''))
    else:
        return float(datarate)
    
    
def sort_by_nodes(item):
    return item[0], item[1]


def draw_unique_numbers(length):
    if length <= 1:
        return 0, 0
    else:
        num1 = random.randint(0, length-1)
        num2 = random.choice([i for i in range(length) if i != num1])
        return num1, num2


def extract_info_and_reformat(text):
    # 定位并提取节点信息
    node_info_start = text.index("from node")
    node_info_end = text.index(" is less than")
    node_info = text[node_info_start:node_info_end]

    # 分割文本以提取各个条件
    parts = text.split(", ")
    base_sentence = "Now we need to ensure that"
    
    # 初始化列表来存储修改后的句子
    formatted_sentences = []

    # 遍历各部分，提取和重构条件句
    for part in parts:
        if "end-to-end delay" in part:
            delay_info = part.split(" is less than ")
            delay = delay_info[1]
            formatted_sentence = f"{base_sentence} the end-to-end delay of the traffic {node_info} is less than {delay}"
            formatted_sentences.append(formatted_sentence)
        elif "average jitter" in part:
            jitter_info = part.split(" is less than ")
            jitter = jitter_info[1]
            formatted_sentence = f"{base_sentence} the average jitter {node_info} is less than {jitter}"
            formatted_sentences.append(formatted_sentence)
        elif "packet loss rate" in part:
            packet_loss_info = part.split(" is less than ")
            packet_loss = packet_loss_info[1]
            formatted_sentence = f"{base_sentence} the packet loss rate {node_info} is less than {packet_loss}"
            formatted_sentences.append(formatted_sentence)

    return formatted_sentences


# 设置你的顶级目录
root_dir = "/home/zy/DatasetGen/bgp/update_data_70"




config_total = []
topology_list = []

query_total = []
label_total = []
edge_in = []
edge_out = []
edge_attr = []
count = 0

file_num = 100
node_num = 70

for folder_num in range(file_num):
    main_folder_path = os.path.join(root_dir, str(folder_num))
    
    # configs_folder_path = os.path.join(main_folder_path, "configs")
    cfg_files = glob.glob(os.path.join(main_folder_path, 'configs', '*.cfg'))
    config_desc = []
    
    # 遍历每个configs文件夹中的.cfg文件
    for cfg_file in cfg_files:
        
        with open(cfg_file, 'r') as f:
            cfg_content = f.read()
            # config_desc.append(cfg_content)
            
            config_total.append(cfg_content)
        
        
    topology_sub = []
    
    # 打开和读取topology.csv
    with open(os.path.join(main_folder_path, 'topology.csv'), 'r') as f:
        next(f)
        for line in f:
            node1, node2, delay, datarate, per = line.strip().split(',')
            # Convert delay and datarate to desired units
            delay_ms = convert_delay(delay)
            datarate_mbps = convert_datarate(datarate)
            per = float(per)
            topology_sub.append([int(node1), int(node2), [delay_ms, datarate_mbps, per]])
    
    topology_sub += [[item[1], item[0], item[2]] for item in topology_sub]
    topology_sub.sort(key=sort_by_nodes)
    topology_list.append(topology_sub)
    
    
    # update_record.pkl
    label_path = os.path.join(main_folder_path, "update_record.pkl")
    with open(label_path, 'rb') as f:
        label = pickle.load(f)
        
    
    
    # 处理每个end2end_qos.jsonl文件
    tt = []
    qos_jsonl_path = os.path.join(main_folder_path, "end2end_qos.jsonl")
    with open(qos_jsonl_path, 'r') as file:
        for line in file:
            # 解析每一行 JSON 对象
            tt.append(json.loads(line))

    
    query_temp = []
    count = 0
    for item in tt:
        query_temp.append(item['qos_constraint'])
        count = count + 1
    
    split_texts_list = [extract_info_and_reformat(text) for text in query_temp]
    
    query_total.append(split_texts_list)
    
    label_total.append(str(label))


print(len(config_total))

labels_split = []
query_tt = []
query_spilt = []
node_ids = []

for i in range(0, file_num * node_num):
    labels_split.append(label_total[i // node_num])
    if i % node_num == 0:
        # labels_split.append(label_total[i // node_num])
        query_tt.append(query_total[i // node_num])
    else:
        # labels_split.append([[]])
        query_tt.append(' ')



print(len(labels_split))
# print(len(query_tt))
# print(query_tt[0])
# print(labels_split[0:20])

count = 0
for folder in range(0, file_num):
    for i in range(0, node_num):
        node_ids.append(i + node_num * count)
    count = count + 1


for i in range(0, file_num * node_num):
    pos = i // node_num
    info = query_total[pos]
    
    rate = random.randint(1,100)
    query_info = ''
    if rate <= 50:
        a, b = draw_unique_numbers(node_num - 4)
        query_info = f'### Now we need to route the plane from node {a} to node 12 to be forward through node {b}.\n'
        a, b = draw_unique_numbers(len(info))
        aa = random.randint(0,2)
        bb = random.randint(0,2)
        query_info = query_info + info[a][aa] + info[b][bb] + ' '
        a, b = draw_unique_numbers(len(info))
        aa = random.randint(0,2)
        bb = random.randint(0,2)
        query_info = query_info + info[a][aa] + info[b][bb] + ' Please provide the changes needed to satisfy the QoS constraints for the configuration.'
    
    else:
        a, b = draw_unique_numbers(len(info))
        aa = random.randint(0,2)
        bb = random.randint(0,2)
        query_info = "### " + info[a][aa] + info[b][bb] + ' '
        
        a, b = draw_unique_numbers(len(info))
        aa = random.randint(0,2)
        bb = random.randint(0,2)
        query_info = query_info + info[a][aa] + info[b][bb] + ' Please provide the changes needed to satisfy the QoS constraints for the configuration.'
    query_spilt.append(query_info)
    
print(query_spilt[0:5])
print(len(query_spilt))
    

# ================ data.arrow =====================
data_total = {
    'node_ids': node_ids,
    'config_desc': config_total,
    'query': query_spilt,
    'labels': labels_split
}

dataset = Dataset.from_dict(data_total)
dataset.save_to_disk('/home/jzt/Temp/dataset/update_config_70/')
df = pd.DataFrame(data_total)

# 将DataFrame转换为Arrow Table
table = pa.Table.from_pandas(df)
# 将Arrow Table保存为.arrow文件
with open('/home/jzt/Temp/dataset/update_config_70/config_data.arrow', 'wb') as f:
    writer = pa.ipc.new_file(f, table.schema)
    writer.write(table)
    writer.close()


# ====================== edge ============================
temp_list = topology_list


# print(temp_list[0])


for i in range(0, len(temp_list)):
    for item in temp_list[i]:
        item[0] = item[0] + i * node_num
        item[1] = item[1] + i * node_num

edge_in = []
edge_out = []
edge_attr = []

for i in range(0, len(temp_list)):
    for item in temp_list[i]:
        edge_in.append(item[0])
        edge_out.append(item[1])
        edge_attr.append(item[2])


edge_index = []
edge_index.append(edge_in)
edge_index.append(edge_out)
edge_index = np.array(edge_index)
edge_index = torch.from_numpy(edge_index)
edge_attr = np.array(edge_attr)
edge_attr = torch.from_numpy(edge_attr)
with open('/home/jzt/Temp/dataset/update_config_70/edge_index.pkl', 'wb') as f:
    pickle.dump(edge_index, f)
with open('/home/jzt/Temp/dataset/update_config_70/edge_attr.pkl', 'wb') as f:
    pickle.dump(edge_attr, f)



# ================ split.pkl =====================
numbers = list(range(0, file_num * node_num))
train_list = []
valid_list = []
test_list = []

# test 
for i in range(0, 1500):
    num = random.choice(numbers)
    numbers.remove(num)
    test_list.append(num)
test_list.sort()

# valid 
for i in range(0, 500):
    num = random.choice(numbers)
    numbers.remove(num)
    valid_list.append(num)
valid_list.sort()

# train 
# train_list = [x * node_num for x in numbers]
train_list = numbers
train_list = np.array(train_list)
valid_list = np.array(valid_list)
test_list = np.array(test_list)

train_list = torch.from_numpy(train_list)
valid_list = torch.from_numpy(valid_list)
test_list = torch.from_numpy(test_list)

data = {
    'train': train_list,
    'valid': valid_list,
    'test': test_list
}

with open('/home/jzt/Temp/dataset/update_config_70/split.pkl', 'wb') as f:
    pickle.dump(data, f)