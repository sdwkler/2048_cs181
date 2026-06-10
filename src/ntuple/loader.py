import mmap
import struct
import numpy as np

def fast_mmap_load(tdl_obj, filepath):
    """全局通用的零拷贝内存映射加载器"""
    f = open(filepath, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    
    offset = 0
    num_feats = struct.unpack_from('Q', mm, offset)[0]
    offset += 8
    
    if num_feats != len(tdl_obj.feats):
        raise ValueError("特征数量与二进制文件不匹配！")
        
    for feat in tdl_obj.feats:
        name_len = struct.unpack_from('I', mm, offset)[0]
        offset += 4
        
        name = mm[offset:offset+name_len].decode('utf-8')
        offset += name_len
        
        weight_size = struct.unpack_from('Q', mm, offset)[0]
        offset += 8
        
        feat.weight = np.ndarray(shape=(weight_size,), dtype=np.float32, buffer=mm, offset=offset)
        offset += weight_size * 4
        
    tdl_obj._mmap_file = f
    tdl_obj._mmap_handle = mm