import mmap
import struct
import numpy as np

def fast_mmap_load(tdl_obj, filepath):
    """
    修复了内存对齐 Bug，并针对纯 Python 循环加速的加载器
    """
    f = open(filepath, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    
    offset = 0
    num_feats = struct.unpack_from('Q', mm, offset)[0]
    offset += 8
    
    if num_feats != len(tdl_obj.feats):
        f.close()
        raise ValueError(f"特征数量不匹配！文件有 {num_feats} 个，模型需要 {len(tdl_obj.feats)} 个")
        
    for feat in tdl_obj.feats:
        # 1. 读取名字长度
        name_len = struct.unpack_from('I', mm, offset)[0]
        offset += 4
        
        # 2. 读取名字
        name = mm[offset:offset+name_len].decode('utf-8')
        offset += name_len
        
        # 3. 读取权重数量
        weight_size = struct.unpack_from('Q', mm, offset)[0]
        offset += 8
        
        byte_length = weight_size * 4
        
        # 4. 【核心修复与提速】
        # 切片 mm 会生成一个新的、天然内存对齐的 bytes 对象
        raw_bytes = mm[offset : offset + byte_length]
        
        # 将 bytes 转为 numpy 数组后，立刻 .tolist() 变为原生列表。
        # 这消除了 Numpy scalar 在 Python 里的封箱开销，让 estimate 速度翻倍。
        feat.weight = np.frombuffer(raw_bytes, dtype=np.float32).tolist()
        
        offset += byte_length
        
    # 由于已经转成了 list 放进内存，不再需要保持底层文件句柄开启了
    mm.close()
    f.close()
    
    # 删掉这两个属性，防止 run_search.py 结尾试图重复关闭导致报错
    if hasattr(tdl_obj, '_mmap_file'):
        delattr(tdl_obj, '_mmap_file')
    if hasattr(tdl_obj, '_mmap_handle'):
        delattr(tdl_obj, '_mmap_handle')