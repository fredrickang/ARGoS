from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import random
import pickle
import sys
from itertools import islice
import math

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 사용 예시
set_seed(22)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
import torch.nn as nn


class MLPBranchingNet(nn.Module):
    def __init__(self, 
                 input_dim,  # state feature dimension (input vector size)
                 action_size_per_branch,  # number of discrete actions per branch
                 num_action_branches,  # number of action branches (action dimensions)
                 hiddens_common=[],  # hidden layer sizes for shared trunk MLP
                 hiddens_actions=[],  # hidden layer sizes for each action head MLP
                 hiddens_value=[],  # hidden layer sizes for state-value MLP (dueling)
                 independent=False,  # if True, each branch has its own trunk MLP
                 dueling=True,  # if True, use dueling architecture (V + A)
                 aggregator='reduceLocalMean',  # advantage aggregator/centering method
                 distributed_single_stream=False):  # reserved for single-stream variant (not implemented)
        super(MLPBranchingNet, self).__init__()

        # 초기 설정 및 검증
        if dueling:
            assert aggregator in ['reduceLocalMean', 'reduceGlobalMean', 'naive', 'reduceLocalMax', 'reduceGlobalMax'], \
                '듀얼링 아키텍처를 사용할 때는 적절한 aggregator 방법이 설정되어야 합니다.'
            assert hiddens_value, '듀얼링 아키텍처를 사용할 때는 state-value 네트워크 레이어 크기가 비어 있을 수 없습니다.'
        else:
            assert aggregator is None or aggregator == 'naive', \
                '듀얼링 아키텍처를 사용하지 않을 때는 aggregator 방법이 설정되지 않아야 합니다.'
            assert not hiddens_value, \
                '듀얼링 아키텍처를 사용하지 않을 때는 state-value 네트워크 레이어 크기가 비어 있어야 합니다.'

        if num_action_branches < 2 and independent:
            raise ValueError('independent는 action dimension이 2개 이상일 때만 의미가 있습니다.')

        if distributed_single_stream:
            raise NotImplementedError("distributed_single_stream=True 케이스는 아직 구현되어 있지 않습니다.")

        self.independent = independent
        self.dueling = dueling
        self.aggregator = aggregator
        self.distributed_single_stream = distributed_single_stream
        self.num_action_branches = num_action_branches
        self.action_size_per_branch = action_size_per_branch
        self.input_dim = input_dim

        # 공유 네트워크 모듈 생성
        if not independent:
            self.shared_layers = self.build_layers(input_dim, hiddens_common)
        else:
            self.indep_shared_layers = nn.ModuleList(
                [self.build_layers(input_dim, hiddens_common) for _ in range(num_action_branches)]
            )

        # 액션 브랜치 생성
        self.action_branches = nn.ModuleList()
        action_output_dim = self.action_size_per_branch

        shared_output_dim = hiddens_common[-1] if hiddens_common else input_dim
        for _ in range(num_action_branches):
            layers = self.build_layers(shared_output_dim, hiddens_actions)
            last_dim = hiddens_actions[-1] if hiddens_actions else shared_output_dim
            layers.add_module('output', nn.Linear(last_dim, action_output_dim))
            self.action_branches.append(layers)

        # 듀얼링 아키텍처의 경우 상태 가치 브랜치 생성
        if dueling:
            if not independent:
                self.state_value_layers = self.build_layers(shared_output_dim, hiddens_value)
                last_dim = hiddens_value[-1] if hiddens_value else shared_output_dim
                self.state_value_layers.add_module('output', nn.Linear(last_dim, 1))
            else:
                self.state_value_branches = nn.ModuleList()
                for _ in range(num_action_branches):
                    layers = self.build_layers(shared_output_dim, hiddens_value)
                    last_dim = hiddens_value[-1] if hiddens_value else shared_output_dim
                    layers.add_module('output', nn.Linear(last_dim, 1))
                    self.state_value_branches.append(layers)

    def build_layers(self, input_dim, hidden_dims):
        layers = nn.Sequential()
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.add_module(f'fc{i}', nn.Linear(prev_dim, hidden_dim))
            layers.add_module(f'relu{i}', nn.ReLU())
            prev_dim = hidden_dim
        return layers

    def forward(self, x):
        """
        고정된 브랜치 개수(self.num_action_branches)만 사용.
        출력 shape: [batch_size, num_action_branches, action_size_per_branch]
        """
        B = x.shape[0]
        branch_count = self.num_action_branches

        # 공유 네트워크 모듈 처리
        if not self.independent:
            out = self.shared_layers(x)  # [B, D]
        else:
            out = [self.indep_shared_layers[i](x) for i in range(branch_count)]  # list of [B, D]

        # 액션 브랜치 계산
        action_scores = []
        if not self.independent:
            for i in range(branch_count):
                action_out = self.action_branches[i](out)  # [B, A]
                action_scores.append(self._apply_adv_aggregator(action_out))
        else:
            for i in range(branch_count):
                action_out = self.action_branches[i](out[i])  # [B, A]
                action_scores.append(self._apply_adv_aggregator(action_out))

        # 듀얼링 결합
        if self.dueling:
            if not self.independent:
                state_value = self.state_value_layers(out)  # [B, 1]
                q_values = self._combine_dueling(state_value, action_scores)
            else:
                q_values = []
                if self.aggregator == 'reduceGlobalMean':
                    # 전역 평균 계산은 모든 브랜치 action_scores가 필요
                    action_scores_concat = torch.cat(action_scores, dim=1)  # [B, branch*A]
                    global_mean = action_scores_concat.mean(1, keepdim=True)  # [B, 1]

                for i in range(branch_count):
                    state_value_i = self.state_value_branches[i](out[i])  # [B, 1]
                    if self.aggregator == 'reduceGlobalMean':
                        a_adj = action_scores[i] - global_mean
                        q_values.append(state_value_i + a_adj)
                    else:
                        q_values.append(state_value_i + action_scores[i])
        else:
            q_values = action_scores

        # [B, branch, A]
        q_values = torch.stack(q_values, dim=1)
        return q_values

    def _apply_adv_aggregator(self, action_out):
        """
        action_out: [B, A]
        return: [B, A] (필요 시 중심화된 advantage 형태)
        """
        if self.aggregator == 'reduceLocalMean':
            m = action_out.mean(1, keepdim=True)
            return action_out - m
        elif self.aggregator == 'reduceLocalMax':
            mx = action_out.max(1, keepdim=True)[0]
            return action_out - mx
        elif self.aggregator in ['naive', 'reduceGlobalMean']:
            return action_out
        elif self.aggregator == 'reduceGlobalMax':
            raise NotImplementedError('reduceGlobalMax는 구현되어 있지 않습니다.')
        else:
            raise ValueError('지원되지 않는 aggregator 방법입니다.')

    def _combine_dueling(self, state_value, action_scores):
        """
        state_value: [B,1]
        action_scores: list of [B,A]
        return: list of [B,A]
        """
        if self.aggregator in ['reduceLocalMean', 'reduceLocalMax', 'naive']:
            return [state_value + a for a in action_scores]
        elif self.aggregator == 'reduceGlobalMean':
            action_scores_concat = torch.cat(action_scores, dim=1)  # [B, branch*A]
            global_mean = action_scores_concat.mean(1, keepdim=True)  # [B,1]
            adjusted = [a - global_mean for a in action_scores]
            return [state_value + a for a in adjusted]
        elif self.aggregator == 'reduceGlobalMax':
            raise NotImplementedError('reduceGlobalMax는 구현되어 있지 않습니다.')
        else:
            raise ValueError('지원되지 않는 aggregator 방법입니다.')


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_dim: int,
        action_bins: int,
        *,
        gamma: float = 0.95,
        lr: float = 1e-2,
        batch_size: int = 32,
        replay_size: int = 50_000,
        target_update_period: int = 10,
        tau: float = 0.005,
        save_path: str = "./outputs/dqn_branching_model.pth",
        load_path: str | None = None,
        hiddens_common=(64, 64),
        hiddens_actions=(32,),
        hiddens_value=(32,),
    ):
        self.state_size = state_size
        self.action_dim = action_dim
        self.action_bins = action_bins
        hiddens_common = list(hiddens_common)
        hiddens_actions = list(hiddens_actions)
        hiddens_value = list(hiddens_value)

        # RL hyperparams (train_model에서 사용)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_period = target_update_period
        self.train_steps = 0
        self.tau = tau

        # Exploration (업데이트 로직이 별도라면 유지)
        self.epsilon = 0.7
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.0

        # Replay buffer (통일된 transition 포맷)
        # (s, a[action_dim], r, s2, done, next_valid_mask)
        self.memory = deque(maxlen=replay_size)

        # Networks
        self.model = MLPBranchingNet(
            input_dim=self.state_size,
            action_size_per_branch=self.action_bins,
            num_action_branches=self.action_dim,
            hiddens_common=hiddens_common,
            hiddens_actions=hiddens_actions,
            hiddens_value=hiddens_value,
            independent=False,
            dueling=True,
            aggregator='reduceLocalMean',
            distributed_single_stream=False,
        ).to(device)

        self.target_model = MLPBranchingNet(
            input_dim=self.state_size,
            action_size_per_branch=self.action_bins,
            num_action_branches=self.action_dim,
            hiddens_common=hiddens_common,
            hiddens_actions=hiddens_actions,
            hiddens_value=hiddens_value,
            independent=False,
            dueling=True,
            aggregator='reduceLocalMean',
            distributed_single_stream=False,
        ).to(device)

        self.update_target_model()

        # Optimizer / loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Save paths
        self.save_path = save_path
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        # Optional load
        if load_path is not None:
            self.load_model(load_path)

    # ---------- I/O ----------
    def save_model(self, policy_path: str | None = None):
        policy_path = policy_path or self.save_path
        os.makedirs(os.path.dirname(policy_path), exist_ok=True)
        torch.save(self.model.state_dict(), policy_path)

    def load_model(self, policy_path: str):
        ckpt = torch.load(policy_path, map_location=device)
        self.model.load_state_dict(ckpt)
        self.target_model.load_state_dict(ckpt)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # ---------- Replay ----------
    def store_transition(
        self,
        state,
        action,
        reward: float,
        next_state,
        done: bool,
    ):
        """
        action: 길이가 action_dim인 액션 벡터라고 가정.
        내부에서 action_dim으로 패딩/절단해서 저장.
        """
        a = np.asarray(action, dtype=np.int64)

        # action_dim 길이 맞추기 (패딩/절단)
        if a.size > self.action_dim:
            a = a[: self.action_dim]
        elif a.size < self.action_dim:
            a = np.pad(a, (0, self.action_dim - a.size), constant_values=0)

        self.memory.append((state, a, float(reward), next_state, bool(done)))

    # ---------- Policy ----------
    @torch.no_grad()
    def get_action(self, state, explore_policy, budget):

        if not isinstance(state, torch.Tensor):
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            state_t = state.to(device)
            if state_t.ndim == 1:
                state_t = state_t.unsqueeze(0)

        # Per-window epsilon decay (optional): decay EVERY window, not only on explore, so
        # epsilon reliably anneals to ~0 after settling. The default on-explore-only decay
        # self-limits (fewer explores -> less decay) and plateaus at a small-but-nonzero
        # epsilon, whose residual random allocations are the dominant steady-state util jitter.
        if getattr(self, "epsilon_decay_per_window", False) and self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # epsilon-greedy exploration
        if np.random.rand() < self.epsilon:
            if not getattr(self, "epsilon_decay_per_window", False):
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                else:
                    self.epsilon = self.epsilon_min

            action = explore_policy(budget)

            return action, True

        q_values = self.model(state_t)     # [1, action_dim, action_bins]
        action = q_values.squeeze(0).argmax(dim=1).cpu().numpy()
        return action, False

    def random_explore_policy(self, budget=None):
        # Random action for exploration; budget is unused.
        return np.random.randint(0, self.action_bins, size=self.action_dim)
    
    
    # ---------- Training ----------
    def compute_reward(
        self,
        util: float,
        qos_list: list[int],
        value: float,
        *,
        U_ref: float,
        Q_max: int,
        w_u: float = 1.0,
        w_v: float = 1.0,
        band_ratio: float = 0.02,
        hard_thr: float = 0.98,
        P_hard: float = 5.0,
        k_hard: float = 100.0,
        eps: float = 1e-8,
    ):
        """
        util: observed utilization in [0, 1]
        qos_list: QoS levels of executed tasks
        """

        # --------------------------------------------------
        # 1) Utilization band reward (±10%)
        # --------------------------------------------------e
        band = band_ratio * U_ref
        
        U_low  = max(U_ref - band, 0.0)
        U_high = min(U_ref + band, hard_thr - eps)
        
        denom = band + eps

        if U_low <= util <= U_high:
            # peak at util == U_ref
            util_reward = 1.0 - ((abs(util - U_ref) / denom) ** 2)
        else:
            # quadratic penalty outside band
            if util < U_low:
                d = U_low - util
            else:
                d = util - U_high
            util_reward = -((d / denom) ** 2)

        # --------------------------------------------------
        # 2) Hard penalty when util >= 0.98
        # --------------------------------------------------
        hard_penalty = 0.0
        if util >= hard_thr:
            hard_penalty = P_hard + k_hard * ((util - hard_thr) ** 2)

        # --------------------------------------------------
        # 3) Value term: sum(qos^2) over executed tasks
        # --------------------------------------------------
        if util > U_high:
            value = 0.0
    
        # --------------------------------------------------
        # 4) Final reward
        # --------------------------------------------------
        
        eps = 1e-6
        reward = w_u*np.log(util_reward+eps) + w_v*np.log(value+eps) - hard_penalty
        
        
        # optional diagnostics
        info = {
            "util_reward": util_reward,
            "value_reward": value,
            "hard_penalty": hard_penalty,
            "util": util,
        }

        return reward, info
    
    
    def compute_reward_decoupled_sum(
        self,
        util: float,
        value: float,
        *,
        U_ref: float,
        dmr: float | None = None,
        stage: str = 'auto',  # 'value_first', 'util_first', 'auto'
        objectives: dict = None,
        band_ratio: float = 0.05,
        hard_thr: float = 0.98,
        eps: float = 1e-8,
        out_range: str = "signed",   # "signed" or "01"
        scale: float = 5.0,
        miss_penalty_base: float = 0.5,
        miss_penalty_scale: float = 2.0,
    ):
        """
        Sum-based multi-objective reward (stage-dependent weights)

        - 기존과 동일하게 score들을 0~1로 만든 뒤,
        Weighted geometric mean(곱) 대신 Weighted sum(합)으로 결합.
        - out_range:
            * "signed": (score-1) 형태로 0 중심(-) 보상 제공 (tracking이 보통 더 잘 됨)
            * "01": 0~1 범위 평균을 그대로 사용
        - scale: 최종 reward 스케일 (기존 코드에서 *10 하던 것 유지 가능)
        """

        if objectives is None:
            objectives = {
                'util_proximity': {'weight': 5.0, 'scale': 1.0},
                'value': {'weight': 1.0, 'scale': 1.0},
                'safety': {'weight': 1.0, 'scale': 1.0},
            }

        scores = {}

        # -----------------------
        # 1) Util proximity score (0~1)
        # -----------------------
        band = max(band_ratio * U_ref, eps)
        U_low = max(U_ref - band, 0.0)
        U_high = min(U_ref + band, hard_thr - eps)

        in_band = (U_low <= util <= U_high)

        if in_band:
            util_score = np.exp(-((util - U_ref) / band) ** 2)
        else:
            if util < U_low:
                d = U_low - util
            else:
                d = util - U_high
            # Outside band: penalize proportionally to distance (cap at 2x band).
            util_score = -min(d / band, 2.0)

        # hard threshold(안전 한계) 근처/초과 패널티
        if util >= hard_thr:
            util_score *= np.exp(-100.0 * (util - hard_thr))

        scores['util_proximity'] = float(np.clip(util_score, -2.0, 1.0))

        # -----------------------
        # 2) Value score (0~1)
        # -----------------------
        if in_band:
            scores['value'] = float(np.clip(value, 0.0, 1.0))
        else:
            # band 밖으로 벗어날수록 value가 음수가 되도록
            if util < U_low:
                d = U_low - util
            else:
                d = util - U_high
                # d가 클수록 음수 값이 커짐
                
            penalty_factor = -d / band
            scores['value'] = float(np.clip(value + penalty_factor, -1.0, 1.0))

        # -----------------------
        # 3) Safety score (0~1)
        # -----------------------
        if util < hard_thr:
            safety_score = 1.0
        else:
            safety_score = np.exp(-10.0 * (util - hard_thr))
        scores['safety'] = float(np.clip(safety_score, 0.0, 1.0))

        # -----------------------
        # Stage별 가중치 (기존 로직 유지)
        # -----------------------
        if stage == 'value_first':
            weights = {'util_proximity': 0.3, 'value': 2.0, 'safety': 1.0}
        elif stage == 'util_first':
            weights = {'util_proximity': 5.0, 'value': 0.5, 'safety': 1.0}
        else:  # 'auto'
            weights = {'util_proximity': 2.0, 'value': 0.5, 'safety': 1.0}

        # -----------------------
        # (핵심) Sum-based aggregation
        # -----------------------
        total_weight = 0.0
        reward_sum = 0.0

        for obj_name, w_stage in weights.items():
            if obj_name not in scores:
                continue

            # objective dict의 weight/scale도 반영 (기존 코드의 objectives 인자 활용)
            obj_w = objectives.get(obj_name, {}).get('weight', 1.0)
            obj_s = objectives.get(obj_name, {}).get('scale', 1.0)

            w = float(w_stage) * float(obj_w)
            s = float(obj_s)

            sc = float(scores[obj_name])
            # out_range="signed"면 0 중심의 (-) penalty 형태로 변환
            if out_range == "signed":
                # score=1이면 0, score<1이면 음수 (tracking에서 더 잘 움직이는 경우가 많음)
                term = (sc - 1.0) * s
            elif out_range == "01":
                term = sc * s
            else:
                raise ValueError("out_range must be 'signed' or '01'")

            reward_sum += w * term
            total_weight += w

        # 정규화 (가중 평균)
        reward = reward_sum / max(total_weight, eps)

        # 스케일링 (기존 *10 유지)
        reward *= scale

        # Hard penalty when deadline miss occurs.
        if dmr is not None and dmr > 0.0:
            reward -= (miss_penalty_base + miss_penalty_scale * dmr) * scale

        info = {
            **scores,
            'reward': reward,
            'util': util,
            'value': value,
            'in_band': in_band,
            'stage': stage,
            'weights': weights,
            'objectives': objectives,
            'out_range': out_range,
            'scale': scale,
            'dmr': dmr,
        }

        return reward, info

    def compute_reward_v2(
        self,
        util: float,
        value: float,
        *,
        U_ref: float,
        dmr: float | None = None,
        sigma_frac: float = 0.10,
        w_u: float = 1.0,
        w_v: float = 1.0,
        hard_thr: float = 0.98,
        sat_penalty: float = 1.0,
        miss_base: float = 0.5,
        miss_scale: float = 2.0,
        scale: float = 5.0,
    ):
        """Positive-capable, smooth reward (fixes the decoupled_sum signal).

        Problems it addresses:
          * decoupled_sum uses out_range='signed' so every term is (score-1) <= 0 -> the
            reward is ALWAYS <= 0, i.e. there is no positive reinforcement to climb toward;
          * its util band is narrow (~+/-3.5%) and it up-weights util 4x over value, so the
            QoS-value signal is weak and mostly gated off.

        This reward:
          * peaks at +scale when util == U_ref AND avg-QoS value is maximal AND no
            saturation/miss -> good behaviour earns a clear POSITIVE signal;
          * tracks util with a smooth Gaussian (sigma = sigma_frac*U_ref, wide enough to
            give gradient over ~+/-2*sigma) instead of a hard band;
          * lets the QoS-value term contribute ALWAYS (balanced w_u:w_v), so the agent is
            always incentivised to raise QoS when the util/safety terms allow it;
          * keeps soft saturation and hard deadline-miss penalties for safety.

        `value` is the execution-weighted average QoS fraction in [0,1] (as built by the
        caller), so maximising it == maximising delivered QoS.
        """
        sigma = max(sigma_frac * U_ref, 1e-3)
        util_score = float(np.exp(-0.5 * ((util - U_ref) / sigma) ** 2))  # (0,1], 1 at ref
        value_score = float(np.clip(value, 0.0, 1.0))                     # avg QoS fraction
        core = (w_u * util_score + w_v * value_score) / (w_u + w_v)       # [0,1]
        reward = scale * core
        if util > hard_thr:  # soft saturation penalty, 0 at hard_thr -> sat_penalty at util=1
            reward -= scale * sat_penalty * ((util - hard_thr) / max(1.0 - hard_thr, 1e-6))
        if dmr is not None and dmr > 0.0:  # deadline-miss penalty
            reward -= (miss_base + miss_scale * dmr) * scale
        info = {"util_score": util_score, "value_score": value_score, "reward": reward}
        return reward, info

    def reward_gate_band(self, util, value, U_ref, *,
                        band_ratio=0.05,
                        hard_thr=0.98,
                        alpha=10.0,      # band 밖 penalty 세기
                        beta=1.0,        # band 안에서 value 보상 세기
                        gamma=50.0,      # hard_thr 초과 시 추가 벌점(선택)
                        eps=1e-8):
        """
        - util이 band 안이면: reward = + beta * value
        - util이 band 밖이면: reward = - alpha * (normalized_error)^2   (value 무시)
        - util이 hard_thr 이상이면 추가 penalty
        """
        band = max(band_ratio * U_ref, eps)
        U_low = max(U_ref - band, 0.0)
        U_high = min(U_ref + band, hard_thr - eps)

        in_band = (U_low <= util <= U_high)

        # hard safety (선택)
        if util >= hard_thr:
            # band 안/밖과 무관하게 강하게 벌점
            return -(gamma + alpha), {"in_band": False, "reason": "hard_thr"}

        if in_band:
            r = beta * float(np.clip(value, 0.0, 1.0))
            return r, {"in_band": True}
        else:
            # band 밖 penalty: 경계까지의 거리만큼 페널티
            if util < U_low:
                d = U_low - util
            else:
                d = util - U_high
            e = d / band  # normalized error
            r = -alpha * (e * e)
            return r, {"in_band": False, "norm_err": e}
        
    def compute_reward_product(
            self,
            util: float,
            value: float,  # 직접 전달받음 (이미 정규화된 값 0~1)
            *,
            U_ref: float,
            objectives: dict = None,
            band_ratio: float = 0.1,
            hard_thr: float = 0.98,
            eps: float = 1e-8,
        ):
        """
        Multi-objective product-based reward
        
        Args:
            util: current utilization [0, 1]
            value: pre-computed value score [0, 1]
            U_ref: reference utilization target
            objectives: dict of objectives with weights and scales
            
        objectives = {
            'util_proximity': {'weight': 1.0, 'scale': 1.0},
            'value': {'weight': 1.0, 'scale': 1.0},
            'safety': {'weight': 1.0, 'scale': 1.0},
        }
        """
        if objectives is None:
            objectives = {
                'util_proximity': {'weight': 1.0, 'scale': 1.0},
                'value': {'weight': 1.0, 'scale': 1.0},
                'safety': {'weight': 1.0, 'scale': 1.0},
            }
        
        scores = {}
        
        # --------------------------------------------------
        # Compute all objective scores [0, 1]
        # --------------------------------------------------
        band = band_ratio * U_ref
        U_low = max(U_ref - band, 0.0)
        U_high = min(U_ref + band, hard_thr - eps)
        
        # Band 내에 있는지 확인
        in_band = U_low <= util <= U_high
        
        # 1) Util proximity
        if 'util_proximity' in objectives:
            if in_band:
                util_score = np.exp(-((util - U_ref) / band) ** 2)
            else:
                if util < U_low:
                    d = U_low - util
                else:
                    d = util - U_high
                util_score = np.exp(-(d / band) ** 2) * 0.5
            
            if util >= hard_thr:
                util_score *= np.exp(-100 * (util - hard_thr))
            
            scale = objectives['util_proximity'].get('scale', 1.0)
            scores['util_proximity'] = np.clip(util_score ** scale, 0.0, 1.0)
        
        # 2) Value (band 내에 있을 때만 의미있게)
        if 'value' in objectives:
            if in_band:
                # Band 내: value를 그대로 사용
                value_score = value
            else:
                # Band 밖: value를 0으로 (또는 매우 작게)
                value_score = 0.0
            
            scale = objectives['value'].get('scale', 1.0)
            scores['value'] = np.clip(value_score ** scale, 0.0, 1.0)
        
        # 3) Safety
        if 'safety' in objectives:
            if util < hard_thr:
                safety_score = 1.0
            else:
                safety_score = np.exp(-10 * (util - hard_thr))
            
            scale = objectives['safety'].get('scale', 1.0)
            scores['safety'] = np.clip(safety_score ** scale, 0.0, 1.0)
        
        # 4) Efficiency (value per util) - band 내에서만 계산
        if 'efficiency' in objectives:
            if in_band and util > eps and value > 0:
                efficiency_score = value / (util / U_ref + eps)
                efficiency_score = min(efficiency_score, 1.0)
            else:
                efficiency_score = 0.0
            
            scale = objectives['efficiency'].get('scale', 1.0)
            scores['efficiency'] = np.clip(efficiency_score ** scale, 0.0, 1.0)
        
        # --------------------------------------------------
        # Weighted geometric mean
        # --------------------------------------------------
        total_weight = sum(obj['weight'] for obj in objectives.values())
        
        product = 1.0
        for obj_name, obj_config in objectives.items():
            if obj_name in scores:
                weight = obj_config['weight']
                score = scores[obj_name] 
                product *= score ** weight
        
        reward = product ** (1.0 / total_weight)
        
        # Scale to [0, 10] for better gradient
        reward = reward * 10.0
        
        info = {**scores, 'reward': reward, 'util': util, 'value': value, 'in_band': in_band}
        
        return reward, info

    def sample_recent_half(self):
        mem = list(self.memory)
        n = len(mem)
        if n == 0:
            return []

        # Use the most recent half of the buffer as the sampling pool.
        recent_count = max(1, n // 2)
        pool = mem[-recent_count:]

        if self.batch_size >= len(pool):
            return pool
        return random.sample(pool, self.batch_size)
    
    def soft_update(self):
        with torch.no_grad():
            for tgt, src in zip(self.target_model.parameters(), self.model.parameters()):
                tgt.data.lerp_(src.data, self.tau)  # tgt ← (1-τ)·tgt + τ·src
    
    def train_model(self):

            
        # -----------------------------
        # 1) minibatch sample
        # -----------------------------
        mini_batch = self.sample_recent_half()
        B = len(mini_batch)
        if B == 0:
            return

        # -----------------------------
        # 2) batch buffers
        # -----------------------------
        states      = np.zeros((B, self.state_size), dtype=np.float32)
        next_states = np.zeros((B, self.state_size), dtype=np.float32)
        actions     = np.zeros((B, self.action_dim), dtype=np.int64)
        rewards     = np.zeros((B,), dtype=np.float32)
        dones       = np.zeros((B,), dtype=np.float32)

        # -----------------------------
        # 3) unpack transitions
        #    허용 포맷: (s,a,r,ns,d) + extra fields는 무시
        # -----------------------------
        for i, tup in enumerate(mini_batch):
            if len(tup) < 5:
                raise ValueError(f"Transition too short: len={len(tup)}")

            s, a, r, ns, d = tup[0], tup[1], tup[2], tup[3], tup[4]

            states[i] = np.asarray(s, dtype=np.float32)
            next_states[i] = np.asarray(ns, dtype=np.float32)

            a = np.asarray(a, dtype=np.int64).reshape(-1)
            if a.size > self.action_dim:
                a = a[:self.action_dim]
            elif a.size < self.action_dim:
                a = np.pad(a, (0, self.action_dim - a.size), constant_values=0)
            actions[i] = a

            rewards[i] = float(r)
            dones[i] = float(d)

        # -----------------------------
        # 4) numpy -> torch
        # -----------------------------
        states      = torch.from_numpy(states).to(device)       # [B, S]
        next_states = torch.from_numpy(next_states).to(device)  # [B, S]
        actions     = torch.from_numpy(actions).to(device)      # [B, A]
        rewards     = torch.from_numpy(rewards).to(device)      # [B]
        dones       = torch.from_numpy(dones).to(device)        # [B]

        # -----------------------------
        # 5) forward pass
        # -----------------------------
        q_values      = self.model(states)               # [B, A, K]
        next_q_online = self.model(next_states)          # [B, A, K]
        next_q_target = self.target_model(next_states)   # [B, A, K]

        # -----------------------------
        # 6) Double DQN target (no masking)
        # -----------------------------
        with torch.no_grad():
            next_actions = next_q_online.argmax(dim=2, keepdim=True)           # [B, A, 1]
            next_q_sel   = next_q_target.gather(2, next_actions).squeeze(2)    # [B, A]
            target_q = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * self.gamma * next_q_sel  # [B, A]

        current_q = q_values.gather(2, actions.unsqueeze(2)).squeeze(2)        # [B, A]
        dqn_loss  = (current_q - target_q).pow(2).mean()

        # -----------------------------
        # 7) small L2 to prevent Q explosion (optional)
        # -----------------------------
        lam_l2 = 1e-5
        l2_reg = q_values.pow(2).mean()
        loss = dqn_loss + lam_l2 * l2_reg

        # -----------------------------
        # 8) optimize
        # -----------------------------
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # -----------------------------
        # 9) target update / save
        # -----------------------------
        self.train_steps += 1
        self.soft_update()

        if (self.train_steps % self.target_update_period) == 0:
            self.save_model()

        return {
            "loss": float(loss.detach().cpu().item()),
            "dqn_loss": float(dqn_loss.detach().cpu().item()),
            "l2_reg": float(l2_reg.detach().cpu().item()),
            "epsilon": float(self.epsilon),
        }
