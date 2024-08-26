import os
import glob
import numpy
import torch
import pickle
import random
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from datasets import Dataset, DatasetDict

# 1000张图 每张图20个节点 
# 数据的根目录
root_dir = '/home/zy/Temp/data'

config_list = []
topology_list = []
A_list = []
q_list = []

def get_two_diff_nums(start, end):
    if end <= start:
        raise ValueError("end must be greater than start!")
    
    num1 = random.randint(start, end)
    num2 = random.randint(start, end)
    
    # 如果两个数相同，重新随机第二个数，直到两个数不同
    while num1 == num2:
        num2 = random.randint(start, end)
    
    return num1, num2


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


# 遍历每个子文件夹
for i in range(300):
    subdir = os.path.join(root_dir, str(i))

    # 打开和读取route_table.txt
    with open(os.path.join(subdir, 'route_table.txt'), 'r') as f:
        # route_table = f.read()
        # print('route_table---{}'.format(i))
        pass

    
    topology_sub = []
    
    A_sub = []
    for ii in range(0, 20):
        A_temp= []
        for jj in range(0, 20):
            A_temp.append(0)
        A_sub.append(A_temp)
    
    # 打开和读取topology.csv
    with open(os.path.join(subdir, 'topology.csv'), 'r') as f:
        next(f)
        for line in f:
            node1, node2, delay, datarate, per = line.strip().split(',')
            # Convert delay and datarate to desired units
            delay_ms = convert_delay(delay)
            datarate_mbps = convert_datarate(datarate)
            per = float(per)
            
            topology_sub.append([int(node1), int(node2), [delay_ms, datarate_mbps, per]])
            A_sub[int(node1)][int(node2)] = [delay_ms, datarate_mbps, per]
            A_sub[int(node2)][int(node1)] = [delay_ms, datarate_mbps, per]
    
    A_list.append(A_sub)
    q_list.append(topology_sub)
    topology_sub += [[item[1], item[0], item[2]] for item in topology_sub]
    topology_sub.sort(key=sort_by_nodes)
    topology_list.append(topology_sub)
    

    config_sub = []
    cfg_files = glob.glob(os.path.join(subdir, 'configs', '*.cfg'))
    for cfg_file in cfg_files:
        # print('config---{}'.format(i))
        with open(cfg_file, 'r') as f:
            cfg_data = f.read()
            config_sub.append(cfg_data)
    config_list.append(config_sub)   


print('Graph Total --- {}'.format(len(config_list)))
print('Node Each Graph --- {}'.format(len(config_list[10])))

# print(len(topology_list))
# print(len(topology_list[1]))
# print(topology_list[1])

# print(len(topology_list))
# print(topology_list[0:10])



# ===================== 邻接矩阵 ========================
# print(A_list[0])



# ===================== data.aarow ======================
# node_ids config_desc query qtype label

# 邻居 时延 传输速率 连接关系
# 每张图总和20条 每个任务5条

node_ids = []
config_desc = []
query = []
qtype = []
labels = []

count = 0
for folder in range(len(topology_list)):
    for i in range(0, 20):
        node_ids.append(i + count * 20)
    count = count + 1
# print(len(node_ids))

for item in config_list:
    for config in item:
        config_desc.append(config)
# print(len(config_desc))

# query --- string
# qtype --- string
# label --- list
for i in range(0, len(topology_list)):
    
    for j in range(0, 20):
        
        qtt = random.randint(1, 5)
        if qtt == 1: # 邻居关系
            node = random.randint(0, 19)
            qq = 'What are the neighbors of network node ' + str(node) + ' ?'
            tt = 'neighbor'
            
            locate = A_list[i]
            neighbor = []
            for pos in range(0, 20):
                if locate[node][pos] != 0:
                    neighbor.append(pos)
            query.append(qq)
            qtype.append(tt)
            labels.append(neighbor)
            
        elif qtt == 2: # 连接关系
            node_s, node_e = get_two_diff_nums(0, 19)
            qq = 'Is there a connection between node ' + str(node_s) + '  and node ' + str(node_e) + ' ?'
            tt = 'connection'
            locate = A_list[i]
            if locate[node_s][node_e] != 0:
                ll = [1]
            else:
                ll = [0]
            query.append(qq)
            qtype.append(tt)
            labels.append(ll)
        
        elif qtt == 3: # 时延
            rate = random.randint(0, 100)
            tt = 'trans delay'
            if rate < 20:
                node_s, node_e = get_two_diff_nums(0, 19)
                qq = "What is the transmission delay between network node " + str(node_s) + '  and node ' + str(node_e) + ' ?'
                
                locate = A_list[i]
                if locate[node_s][node_e] == 0:
                    ll = [0]
                else:
                    info = locate[node_s][node_e]
                    ll = [info[0]]
            else:
                locate = q_list[i]
                pos = random.randint(0, len(locate) - 1)
                info = locate[pos]
                qq = "What is the transmission delay between network node " + str(info[0]) + '  and node ' + str(info[1]) + ' ?'
                ll = [info[2][0]]
                
            query.append(qq)
            qtype.append(tt)
            labels.append(ll)
            
        
        elif qtt == 4: # 传输速率
            
            rate = random.randint(0, 100)
            tt = 'trans rate'
            
            if rate < 20:
                node_s, node_e = get_two_diff_nums(0, 19)
                qq = "What is the transmission rate between network node " + str(node_s) + '  and node ' + str(node_e) + ' ?'
                
                locate = A_list[i]
                if locate[node_s][node_e] == 0:
                    ll = [0]
                else:
                    info = locate[node_s][node_e]
                    ll = [info[1]]
            else:
                locate = q_list[i]
                pos = random.randint(0, len(locate) - 1)
                info = locate[pos]
                qq = "What is the transmission rate between network node " + str(info[0]) + '  and node ' + str(info[1]) + ' ?'
                ll = [info[2][1]]
                
                
            query.append(qq)
            qtype.append(tt)
            labels.append(ll)
            
        else:
            tt = 'forward path'
            node_s, node_e = get_two_diff_nums(0, 19)
            qq = "What is the forward path from network node " + str(node_s) + '  and node ' + str(node_e) + ' ?'
            a,b = get_two_diff_nums(0, 19)
            ll = []
            ll.append(a)
            ll.append(b)
            c = random.randint(0, 19)
            ll.append(c)
            
            query.append(qq)
            qtype.append(tt)
            labels.append(ll)
            
        
print(len(query))
print(len(qtype))
print(len(labels))

data = {
    'node_ids': node_ids,
    'config_desc': config_desc,
    'query': query,
    'qtype': qtype,
    'labels': labels
}
dataset = Dataset.from_dict(data)
dataset.save_to_disk('/home/jzt/Temp/dataset/config_update/')
df = pd.DataFrame(data)

# 将DataFrame转换为Arrow Table
table = pa.Table.from_pandas(df)
# 将Arrow Table保存为.arrow文件
with open('/home/jzt/Temp/dataset/config_update/config_data.arrow', 'wb') as f:
    writer = pa.ipc.new_file(f, table.schema)
    writer.write(table)
    writer.close()
    
    
    

# ====================== edge ============================

temp_list = topology_list

for i in range(0, len(temp_list)):
    for item in temp_list[i]:
        item[0] = item[0] + i * 20
        item[1] = item[1] + i * 20

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
edge_index = numpy.array(edge_index)
edge_index = torch.from_numpy(edge_index)
edge_attr = numpy.array(edge_attr)
edge_attr = torch.from_numpy(edge_attr)
with open('/home/jzt/Temp/dataset/config_update/edge_index.pkl', 'wb') as f:
    pickle.dump(edge_index, f)
with open('/home/jzt/Temp/dataset/config_update/edge_attr.pkl', 'wb') as f:
    pickle.dump(edge_attr, f)
    
    
    
    
# ================ split.pkl =====================
numbers = list(range(0, 6000))
train_list = []
valid_list = []
test_list = []

# test 5000
for i in range(0, 800):
    num = random.choice(numbers)
    numbers.remove(num)
    test_list.append(num)
test_list.sort()

# valid 1000
for i in range(0, 200):
    num = random.choice(numbers)
    numbers.remove(num)
    valid_list.append(num)
valid_list.sort()

# train 5000
train_list = numbers

train_list = numpy.array(train_list)
valid_list = numpy.array(valid_list)
test_list = numpy.array(test_list)

train_list = torch.from_numpy(train_list)
valid_list = torch.from_numpy(valid_list)
test_list = torch.from_numpy(test_list)

data = {
    'train': train_list,
    'valid': valid_list,
    'test': test_list
}

with open('/home/jzt/Temp/dataset/config_update/split.pkl', 'wb') as f:
    pickle.dump(data, f)