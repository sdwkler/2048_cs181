# src/ntuple/loader.py
import mmap
import struct

def fast_mmap_load(tdl_obj, filepath):
    """
    终极零拷贝加载器：利用操作系统底层机制，实现多进程共享同一块物理内存
    """
    # 保持文件打开，绝不能用 with，否则函数结束内存就断开了
    f = open(filepath, "rb")
    
    # 创建只读的内存映射。Windows 保证多进程读取同一文件时，物理内存只占用 1 份！
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    
    offset = 0
    num_feats = struct.unpack_from('Q', mm, offset)[0]
    offset += 8
    
    if num_feats != len(tdl_obj.feats):
        raise ValueError(f"特征数量不匹配！文件有 {num_feats} 个，模型需要 {len(tdl_obj.feats)} 个")
        
    for feat in tdl_obj.feats:
        name_len = struct.unpack_from('I', mm, offset)[0]
        offset += 4
        
        name = mm[offset:offset+name_len].decode('utf-8')
        offset += name_len
        
        weight_size = struct.unpack_from('Q', mm, offset)[0]
        offset += 8
        byte_length = weight_size * 4
        
        # 【全场核心】：使用 memoryview 直接框选内存并转化为 float 视图
        # 这一步没有任何 List 内存分配！PyPy 的 JIT 可以极速读取它！
        feat.weight = memoryview(mm)[offset : offset + byte_length].cast('f')
        
        offset += byte_length
        
    # 将系统句柄挂载到对象上，防止 Python 垃圾回收器把共享内存销毁
    tdl_obj._mmap_file = f
    tdl_obj._mmap_handle = mm