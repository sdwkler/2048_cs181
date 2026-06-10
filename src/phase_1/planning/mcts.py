# src/phase_1/planning/mcts.py
import math
import numpy as np
from src.environments.base_env import board

class MCTSAgent:
    """搭载估值方差与策略熵探针的 MCTS 引擎"""
    def __init__(self, use_afterstate=False, value_func=None, exploration_c=20000.0):
        self.use_afterstate = use_afterstate
        self.exploration_c = exploration_c 
        self.value_func = value_func
        self.Q, self.N, self.N_s, self.visited = {}, {}, {}, set()

    def get_best_action(self, b: board, num_simulations=100):
        self.Q.clear(); self.N.clear(); self.N_s.clear(); self.visited.clear()
        for _ in range(num_simulations): self._simulate(b.raw)
            
        legal_actions = self._get_legal_actions(b.raw)
        if not legal_actions: return 0, 0.0, 0.0
        
        action_scores = []
        action_visits = []
        best_action, best_score = -1, -float('inf')
        
        # 收集根节点数据用于探针计算
        for a in legal_actions:
            after_raw, r = self._apply_action(b.raw, a)
            if self.use_afterstate:
                avg_future = self.Q.get(after_raw, 0) / max(1, self.N.get(after_raw, 0))
                score = r + avg_future
                visits = self.N.get(after_raw, 0)
            else:
                key = (b.raw, a)
                score = self.Q.get(key, 0) / max(1, self.N.get(key, 0))
                visits = self.N.get(key, 0)
                
            action_scores.append(score)
            action_visits.append(visits)
            if score > best_score:
                best_score, best_action = score, a
                
        # 【探针1】：根节点动作估值标准差
        val_variance = np.std(action_scores) if action_scores else 0.0
        
        # 【探针2】：策略访问熵 (反映稳定性)
        total_v = sum(action_visits)
        probs = [v/total_v for v in action_visits] if total_v > 0 else [1/len(legal_actions)]*len(legal_actions)
        policy_entropy = -sum(p * math.log(p + 1e-9) for p in probs if p > 0)
                
        return (best_action if best_action != -1 else legal_actions[0]), val_variance, policy_entropy

    def _simulate(self, state_raw):
        legal_actions = self._get_legal_actions(state_raw)
        if not legal_actions: return 0 
        
        self.N_s[state_raw] = self.N_s.get(state_raw, 0) + 1
        n_s = self.N_s[state_raw]
            
        best_a, best_ucb, best_after_raw, best_r = -1, -float('inf'), None, 0
        for a in legal_actions:
            after_raw, r = self._apply_action(state_raw, a)
            if self.use_afterstate:
                key = after_raw
                n_key = self.N.get(key, 0)
                ucb = float('inf') if n_key == 0 else (r + self.Q.get(key,0)/n_key) + self.exploration_c * math.sqrt(math.log(n_s)/n_key)
            else:
                key = (state_raw, a)
                n_key = self.N.get(key, 0)
                ucb = float('inf') if n_key == 0 else (self.Q.get(key,0)/n_key) + self.exploration_c * math.sqrt(math.log(n_s)/n_key)
                
            if ucb > best_ucb: best_ucb, best_a, best_after_raw, best_r = ucb, a, after_raw, r
                
        key = best_after_raw if self.use_afterstate else (state_raw, best_a)

        if key not in self.visited:
            self.visited.add(key)
            if self.use_afterstate:
                v = self.value_func(board(best_after_raw))
                self.Q[key] = self.Q.get(key,0) + v
                self.N[key] = self.N.get(key,0) + 1
                return best_r + v
            else:
                next_raw = self._apply_popup(best_after_raw)
                v = self.value_func(board(next_raw))
                self.Q[key] = self.Q.get(key,0) + (best_r + v)
                self.N[key] = self.N.get(key,0) + 1
                return best_r + v

        next_raw = self._apply_popup(best_after_raw)
        q_next = self._simulate(next_raw)
        
        if self.use_afterstate:
            self.Q[key] = self.Q.get(key,0) + q_next
        else:
            self.Q[key] = self.Q.get(key,0) + (best_r + q_next)
        self.N[key] = self.N.get(key,0) + 1
        return best_r + q_next

    def _get_legal_actions(self, r): return [a for a in range(4) if board(r).move(a) != -1]
    def _apply_action(self, raw, a): t = board(raw); r = t.move(a); return t.raw, r
    def _apply_popup(self, raw): t = board(raw); t.popup(); return t.raw