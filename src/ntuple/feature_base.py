import struct
import typing
import abc
import math
import random
from src.environments.base_env import board, info, error, debug

class feature(abc.ABC):
    def __init__(self, length : int):
        self.weight = feature.alloc(length)

    def __getitem__(self, i : int) -> float:
        return self.weight[i]

    def __setitem__(self, i : int, v : float) -> None:
        self.weight[i] = v

    def __len__(self) -> int:
        return len(self.weight)

    def size(self) -> int:
        return len(self.weight)

    @abc.abstractmethod
    def estimate(self, b : board) -> float:
        pass

    @abc.abstractmethod
    def update(self, b : board, u : float) -> float:
        pass

    @abc.abstractmethod
    def name(self) -> str:
        pass

    def dump(self, b : board, out : typing.Callable = info) -> None:
        out(f"{b}\nestimate = {self.estimate(b)}")

    def write(self, output : typing.BinaryIO) -> None:
        name = self.name().encode('utf-8')
        output.write(struct.pack('I', len(name)))
        output.write(name)
        size = len(self.weight)
        output.write(struct.pack('Q', size))
        output.write(struct.pack(f'{size}f', *self.weight))

    def read(self, input : typing.BinaryIO) -> None:
        size = struct.unpack('I', input.read(4))[0]
        name = input.read(size).decode('utf-8')
        if name != self.name():
            error(f'unexpected feature: {name} ({self.name()} is expected)')
            exit(1)
        size = struct.unpack('Q', input.read(8))[0]
        if size != len(self.weight):
            error(f'unexpected feature size {size} for {self.name()} ({self.size()} is expected)')
            exit(1)
        self.weight = list(struct.unpack(f'{size}f', input.read(size * 4)))
        if len(self.weight) != size:
            error('unexpected end of binary')
            exit(1)

    @staticmethod
    def alloc(num : int) -> list[float]:
        if not hasattr(feature.alloc, "total"):
            feature.alloc.total = 0
            feature.alloc.limit = (2 << 30) // 4 
        try:
            feature.alloc.total += num
            if feature.alloc.total > feature.alloc.limit:
                raise MemoryError("memory limit exceeded")
            return [float(0)] * num
        except MemoryError as e:
            error("memory limit exceeded")
            exit(-1)
        return None

class pattern(feature):
    def __init__(self, patt : list[int], iso : int = 8):
        super().__init__(1 << (len(patt) * 4))
        if not patt:
            error("no pattern defined")
            exit(1)

        self.isom = [None] * iso
        for i in range(iso):
            idx = board(0xfedcba9876543210)
            if i >= 4:
                idx.mirror()
            idx.rotate(i)
            self.isom[i] = [idx.at(t) for t in patt]

    def estimate(self, b : board) -> float:
        value = 0.0
        # 【提速】：使用局部变量引用 self.weight 减少属性查找
        w = self.weight
        for iso in self.isom:
            value += w[self.indexof(iso, b)]
        return value

    def update(self, b : board, u : float) -> float:
        adjust = u / len(self.isom)
        value = 0.0
        w = self.weight
        for iso in self.isom:
            index = self.indexof(iso, b)
            w[index] += adjust
            value += w[index]
        return value

    def name(self) -> str:
        return f"{len(self.isom[0])}-tuple pattern {self.nameof(self.isom[0])}"

    def dump(self, b : board, out : typing.Callable = info) -> None:
        for iso in self.isom:
            index = self.indexof(iso, b)
            tiles = [(index >> (4 * i)) & 0x0f for i in range(len(iso))]
            out(f"#{self.nameof(iso)}[{self.nameof(tiles)}] = {self[index]}")

    def indexof(self, patt : list[int], b : board) -> int:
        b_at = b.at # 【提速】：局部缓存方法
        # 【提速极限】：针对我们的 6-tuple 进行循环暴力展开
        if len(patt) == 6:
            return b_at(patt[0]) | (b_at(patt[1]) << 4) | (b_at(patt[2]) << 8) | \
                   (b_at(patt[3]) << 12) | (b_at(patt[4]) << 16) | (b_at(patt[5]) << 20)
        
        # 安全后备方案
        index = 0
        for i, pos in enumerate(patt):
            index |= b_at(pos) << (4 * i)
        return index

    def nameof(self, patt : list[int]) -> str:
        return "".join([f"{p:x}" for p in patt])

class diff_pattern(pattern):
    def __init__(self, patt : list[int], iso : int = 8):
        feature.__init__(self, 1 << (5 * (len(patt) - 1)))
        if not patt or len(patt) < 2:
            error("diff pattern requires at least 2 tiles")
            exit(1)

        self.isom = [None] * iso
        for i in range(iso):
            idx = board(0xfedcba9876543210)
            if i >= 4:
                idx.mirror()
            idx.rotate(i)
            self.isom[i] = [idx.at(t) for t in patt]

    def name(self) -> str:
        return f"{len(self.isom[0])}-tuple diff_pattern {self.nameof(self.isom[0])}"

    def indexof(self, patt : list[int], b : board) -> int:
        b_at = b.at # 【提速】
        # 【提速极限】：消除差分计算的 for 循环，直接硬核展开并合并常数计算
        if len(patt) == 6:
            return (b_at(patt[1]) - b_at(patt[0]) + 15) | \
                   ((b_at(patt[2]) - b_at(patt[1]) + 15) << 5) | \
                   ((b_at(patt[3]) - b_at(patt[2]) + 15) << 10) | \
                   ((b_at(patt[4]) - b_at(patt[3]) + 15) << 15) | \
                   ((b_at(patt[5]) - b_at(patt[4]) + 15) << 20)
                   
        index = 0
        for i in range(1, len(patt)):
            index |= (b_at(patt[i]) - b_at(patt[i-1]) + 15) << (5 * (i - 1))
        return index

class move:
    # 保持原样不动
    def __init__(self, board : board = None, opcode : int = -1):
        self.before = None
        self.after = None
        self.opcode = opcode
        self.score = -1
        self.esti = -float('inf')
        if board is not None:
            self.assign(board)

    def state(self) -> board: return self.before
    def afterstate(self) -> board: return self.after
    def value(self) -> float: return self.esti
    def reward(self) -> int: return self.score
    def action(self) -> int: return self.opcode
    def set_state(self, state : board) -> None: self.before = state
    def set_afterstate(self, state : board) -> None: self.after = state
    def set_value(self, value : float) -> None: self.esti = value
    def set_reward(self, reward : int) -> None: self.score = reward
    def set_action(self, action : int) -> None: self.opcode = action

    def assign(self, b : board) -> bool:
        self.after = board(b)
        self.before = board(b)
        self.score = self.after.move(self.opcode)
        self.esti = self.score if self.score != -1 else -float('inf')
        return self.score != -1

    def is_valid(self) -> bool:
        if math.isnan(self.esti):
            error("numeric exception")
            exit(-1)
        return self.after != self.before and self.opcode != -1 and self.score != -1

class learning:
    def __init__(self):
        self.feats = []
        self.scores = []
        self.maxtile = []

    def add_feature(self, feat : feature) -> None:
        self.feats.append(feat)
        sign = f"{feat.name()}, size = {feat.size()}"
        usage = feat.size() * 4
        if usage >= (1 << 30): size = f"{(usage >> 30)}GB"
        elif usage >= (1 << 20): size = f"{(usage >> 20)}MB"
        elif usage >= (1 << 10): size = f"{(usage >> 10)}KB"
        info(f"{sign} ({size})")

    def estimate(self, b : board) -> float:
        # 【提速】：去除 sum() 生成器开销
        v = 0.0
        for feat in self.feats:
            v += feat.estimate(b)
        return v

    def update(self, b : board, u : float) -> float:
        adjust = u / len(self.feats)
        v = 0.0
        for feat in self.feats:
            v += feat.update(b, adjust)
        return v

    def select_best_move(self, b : board) -> move:
        best = move(b)
        # 【提速】：按需创建 move 对象，避免内存分配浪费
        for opcode in range(4):
            mv = move(b, opcode)
            if mv.is_valid():
                mv.set_value(mv.reward() + self.estimate(mv.afterstate()))
                if mv.value() > best.value():
                    best = mv
        return best

    def learn_from_episode(self, path : list[move], alpha : float = 0.1) -> None:
        target = 0
        path.pop()
        while path:
            mv = path.pop()
            error = target - self.estimate(mv.afterstate())
            target = mv.reward() + self.update(mv.afterstate(), alpha * error)

    def make_statistic(self, n : int, b : board, score : int, unit : int = 1000) -> None:
        self.scores.append(score)
        self.maxtile.append(max(b.at(i) for i in range(16)))

        if n % unit == 0:
            avg_score = sum(self.scores) / len(self.scores)
            info(f"{n}\tavg = {avg_score}\tmax = {max(self.scores)}")
            stat = [ self.maxtile.count(i) for i in range(16) ]
            t, c, coef = 1, 0, 100 / unit
            while c < unit:
                if stat[t] != 0:
                    accu = sum(stat[t:])
                    winrate = accu * coef
                    share = stat[t] * coef
                    info(f"\t{(1 << t) & -2}\t{winrate:.1f}%\t({share:.1f}%)")
                c += stat[t]
                t += 1
            self.scores.clear()
            self.maxtile.clear()

    def load(self, path : str) -> None:
        try:
            with open(path, 'rb') as input:
                size = struct.unpack('Q', input.read(8))[0]
                for feat in self.feats:
                    feat.read(input)
        except FileNotFoundError: pass

    def save(self, path : str) -> None:
        with open(path, 'wb') as output:
            output.write(struct.pack('Q', len(self.feats)))
            for feat in self.feats:
                feat.write(output)