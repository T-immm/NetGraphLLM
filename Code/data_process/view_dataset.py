import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

# 用于读取.arrow文件的函数
def read_arrow_file(file_path):
    reader = pa.ipc.open_file(file_path)
    table = reader.read_all()
    return table

# file_path = "/home/jzt/Temp/dataset/update_config/config_data.arrow"
file_path = "/home/jzt/Temp/dataset/new_update_config_14/config_data.arrow"

arrow_table = read_arrow_file(file_path)

df = arrow_table.to_pandas()
# data = df.to_dict()
# 打印DataFrame
print(df)
print(df['query'][10])
data = df.to_dict('list')
# print(data['config_desc'])
