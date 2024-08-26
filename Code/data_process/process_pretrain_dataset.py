import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from datasets import Dataset

def read_arrow_file(file_path):
    reader = pa.ipc.open_file(file_path)
    table = reader.read_all()
    return table

file_path = "/home/jzt/Temp/dataset/config/config_data_20000.arrow"

arrow_table = read_arrow_file(file_path)

df = arrow_table.to_pandas()
# data = df.to_dict()
# 打印DataFrame
print(df)
data = df.to_dict('list')


pretrain_data = {
    'node_ids': data['node_ids'],
    'config_desc': data['config_desc']
}

print(len(pretrain_data['config_desc']))
dataset = Dataset.from_dict(data)
dataset.save_to_disk('/home/jzt/Temp/dataset/pretrain/')

df = pd.DataFrame(pretrain_data)

# 将DataFrame转换为Arrow Table
table = pa.Table.from_pandas(df)

# 将Arrow Table保存为.arrow文件
with open('/home/jzt/Temp/dataset/pretrain/config_data_20000.arrow', 'wb') as f:
    writer = pa.ipc.new_file(f, table.schema)
    writer.write(table)
    writer.close()