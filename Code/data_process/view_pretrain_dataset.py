import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

# 用于读取.arrow文件的函数
def read_arrow_file(file_path):
    reader = pa.ipc.open_file(file_path)
    table = reader.read_all()
    return table

file_path = "/home/jzt/Temp/dataset/pretrain/config_data_20000.arrow"

arrow_table = read_arrow_file(file_path)

df = arrow_table.to_pandas()
# data = df.to_dict()
# 打印DataFrame
print(df)
data = df.to_dict('list')