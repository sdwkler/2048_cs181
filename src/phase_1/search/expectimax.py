# src/phase_1/search/expectimax.py
import src.environments.base_env as env_module

class ExpectimaxAgent:
    """搭载拓扑探针的 Expectimax 引擎"""
    def __init__(self, use_afterstate=False, value_func=None, prune_threshold=0.0001):
        self.use_afterstate = use_afterstate 
        self.prune_threshold = prune_threshold  
        if value_func is None:
            raise ValueError("必须传入 value_func")
        self.value_func = value_func
        
        self.transposition_table = {}
        self.total_nodes_expanded = 0  # 【探针】：用于计算置换表压缩率

    def get_best_action(self, b: env_module.board, max_depth=3):
        self.transposition_table.clear()
        self.total_nodes_expanded = 0
        best_val, best_action = -float('inf'), -1
        
        for action in range(4):
            next_b = env_module.board(b.raw)
            reward = next_b.move(action)
            if reward != -1:
                val = reward + self._chance_node(next_b, max_depth, 1.0)
                if val > best_val:
                    best_val, best_action = val, action

        if best_action == -1:
            for action in range(4):
                next_b = env_module.board(b.raw)
                if next_b.move(action) != -1: return action
            return 0 
            
        # 返回动作以及本步的压缩率 (哈希表唯一节点数 / 总展开节点数)
        compression_ratio = len(self.transposition_table) / max(1, self.total_nodes_expanded)
        return best_action, compression_ratio

    def _chance_node(self, afterstate: env_module.board, depth, current_prob):
        self.total_nodes_expanded += 1
        if current_prob < self.prune_threshold: return self.value_func(afterstate)
            
        tt_key = (afterstate.raw, depth, 'c')
        if tt_key in self.transposition_table: return self.transposition_table[tt_key]

        if self.use_afterstate and depth == 1:
            val = self.value_func(afterstate)
            self.transposition_table[tt_key] = val
            return val
            
        empties = [i for i in range(16) if afterstate.at(i) == 0]
        if not empties:
            val = self.value_func(afterstate)
            self.transposition_table[tt_key] = val
            return val
            
        expected_value, weight = 0, 1.0 / len(empties) 
        for pos in empties:
            afterstate.set(pos, 1) # 模拟 2
            if not self.use_afterstate and depth == 1:
                expected_value += weight * 0.9 * self.value_func(afterstate)
            else:
                expected_value += weight * 0.9 * self._max_node(afterstate, depth - 1, current_prob * weight * 0.9)
            
            afterstate.set(pos, 2) # 模拟 4
            if not self.use_afterstate and depth == 1:
                expected_value += weight * 0.1 * self.value_func(afterstate)
            else:
                expected_value += weight * 0.1 * self._max_node(afterstate, depth - 1, current_prob * weight * 0.1)
                
            afterstate.set(pos, 0)
            
        self.transposition_table[tt_key] = expected_value
        return expected_value

    def _max_node(self, b: env_module.board, depth, current_prob):
        self.total_nodes_expanded += 1
        tt_key = (b.raw, depth, 'm')
        if tt_key in self.transposition_table: return self.transposition_table[tt_key]

        best_val, is_terminal = -float('inf'), True
        for action in range(4):
            next_b = env_module.board(b.raw)
            reward = next_b.move(action)
            if reward != -1:
                is_terminal = False
                val = reward + self._chance_node(next_b, depth, current_prob)
                if val > best_val: best_val = val
                    
        val = self.value_func(b) if is_terminal else best_val
        self.transposition_table[tt_key] = val
        return val