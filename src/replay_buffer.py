"""Prioritized replay buffer for RL-GenRisk hybrid6_raw training.

The node feature matrix is static across the whole training run, while the
action mask changes after selecting a gene.  The static state is therefore
stored once; each transition stores:

    (action, reward, true_current_mask, true_next_mask, history_current_mask,
     history_next_mask, done, preference)

The public method names remain compatible with DQN.py.  ``sample_buffer`` also
returns a final preference batch; for ordinary scalar runs it has shape
``(batch_size, 0)`` and is ignored by DQN.
"""

from __future__ import annotations

import numpy as np


class PrioritizedReplayBuffer:
    """Array-backed proportional prioritized experience replay buffer."""

    def __init__(
        self,
        max_size,
        n_actions,
        feature_dim=3,
        alpha=0.2,
        beta_start=0.1,
        beta_frames=2_000_000,
        eps=1e-5,
        preference_dim=0,
        reward_dim=1,
    ):
        self.mem_size = self._validate_positive_int("max_size", max_size)
        self.n_actions = self._validate_positive_int("n_actions", n_actions)
        self.feature_dim = self._validate_positive_int("feature_dim", feature_dim)
        self.preference_dim = int(preference_dim)
        if self.preference_dim < 0:
            raise ValueError(f"preference_dim must be non-negative, got {preference_dim!r}")
        self.reward_dim = self._validate_positive_int("reward_dim", reward_dim)

        self.alpha = self._validate_unit_interval("alpha", alpha, include_zero=True)
        self.beta_start = self._validate_unit_interval(
            "beta_start", beta_start, include_zero=True
        )
        self.beta_frames = self._validate_positive_int("beta_frames", beta_frames)
        self.eps = float(eps)
        if not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError(f"eps must be a finite positive value, got {eps!r}")

        self.mem_cntr = 0
        self.sample_step = 0

        # 节点特征矩阵在整个训练期间都是静态的，只有动作掩码随选择变化。
        # 因此只保存一份共享状态，逐条 transition 只存 action/reward/mask/done。
        self.state_s = np.zeros(
            (self.n_actions, self.feature_dim),
            dtype=np.float32,
        )
        self.memory_a = np.zeros((self.mem_size, 1), dtype=np.int64)
        self.memory_r = np.zeros((self.mem_size, self.reward_dim), dtype=np.float32)

        # Mask before action: 1=selectable, 0=unavailable/already selected.
        self.memory_ai = np.zeros(
            (self.mem_size, self.n_actions),
            dtype=np.float32,
        )
        # Mask after action.  DQN.py must use this mask for DDQN next-action
        # selection/evaluation instead of reconstructing it implicitly.
        self.memory_sa = np.zeros(
            (self.mem_size, self.n_actions),
            dtype=np.float32,
        )
        # Context masks visible to Q_Fun.  In normal DDQN they exactly equal
        # the true masks above.  History ablations can replace only these
        # arrays while action legality and rewards still use memory_ai/sa.
        self.memory_history_ai = np.zeros(
            (self.mem_size, self.n_actions),
            dtype=np.float32,
        )
        self.memory_history_sa = np.zeros(
            (self.mem_size, self.n_actions),
            dtype=np.float32,
        )
        self.memory_done = np.zeros((self.mem_size, 1), dtype=np.float32)
        self.memory_preference = np.zeros((self.mem_size, self.preference_dim), dtype=np.float32)
        self.priorities = np.zeros((self.mem_size,), dtype=np.float32)

    @staticmethod
    def _validate_positive_int(name, value):
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        return value

    @staticmethod
    def _validate_unit_interval(name, value, include_zero):
        value = float(value)
        lower_ok = value >= 0.0 if include_zero else value > 0.0
        if not np.isfinite(value) or not lower_ok or value > 1.0:
            bracket = "[0, 1]" if include_zero else "(0, 1]"
            raise ValueError(f"{name} must be in {bracket}, got {value!r}")
        return value

    @staticmethod
    def _as_finite_array(name, value, dtype, expected_shape):
        array = np.asarray(value, dtype=dtype)
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} shape must be {expected_shape}, got {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf.")
        return array

    def _validate_mask(self, name, mask):
        mask = self._as_finite_array(
            name,
            mask,
            np.float32,
            (self.n_actions,),
        )
        if not np.all((mask == 0.0) | (mask == 1.0)):
            invalid = np.unique(mask[(mask != 0.0) & (mask != 1.0)])[:10]
            raise ValueError(
                f"{name} must contain only 0/1 values; invalid examples={invalid.tolist()}"
            )
        return mask

    def store_transition(
        self,
        state,
        action,
        reward,
        action_index,
        sel_action,
        done=False,
        preference=None,
        history_mask=None,
        next_history_mask=None,
    ):
        """Store one transition.

        Parameters keep the legacy names for compatibility:
        - action_index: action mask before selecting ``action``.
        - sel_action: action mask after selecting ``action``.
        """
        state = self._as_finite_array(
            "state",
            state,
            np.float32,
            (self.n_actions, self.feature_dim),
        )

        action_value = int(np.asarray(action).reshape(-1)[0])
        if action_value < 0 or action_value >= self.n_actions:
            raise IndexError(
                f"action must be in [0, {self.n_actions - 1}], got {action_value}."
            )

        reward_value = np.asarray(reward, dtype=np.float32).reshape(-1)
        if reward_value.shape != (self.reward_dim,) or not np.isfinite(reward_value).all():
            raise ValueError(
                f"reward must be a finite vector with shape ({self.reward_dim},), got {reward_value.shape}."
            )

        current_action_mask = self._validate_mask("action_index", action_index)
        next_action_mask = self._validate_mask("sel_action", sel_action)
        # Defaults preserve the exact legacy/full-history behavior for all
        # existing callers, including synthetic diagnostics.
        history_current_mask = self._validate_mask(
            "history_mask", current_action_mask if history_mask is None else history_mask
        )
        history_next_mask = self._validate_mask(
            "next_history_mask", next_action_mask if next_history_mask is None else next_history_mask
        )

        if current_action_mask[action_value] != 1.0:
            raise ValueError(
                "Selected action was not available in current_action_mask: "
                f"action={action_value}, mask_value={current_action_mask[action_value]}"
            )
        if next_action_mask[action_value] != 0.0:
            raise ValueError(
                "Selected action must be unavailable in next_action_mask: "
                f"action={action_value}, mask_value={next_action_mask[action_value]}"
            )

        done_value = float(done)
        if done_value not in (0.0, 1.0):
            raise ValueError(f"done must be 0/1 or bool, got {done!r}")
        if self.preference_dim:
            preference_value = self._as_finite_array(
                "preference", preference, np.float32, (self.preference_dim,)
            )
        elif preference is not None:
            preference_value = np.asarray(preference, dtype=np.float32).reshape(-1)
            if preference_value.size:
                raise ValueError("preference was provided but preference_dim is 0.")
            preference_value = preference_value

        index = self.mem_cntr % self.mem_size
        # 状态（节点特征矩阵）在训练中恒定，只存一份副本即可（copy 防止与外部特征矩阵互为别名）。
        self.state_s = state.copy()
        self.memory_a[index, 0] = action_value
        self.memory_r[index] = reward_value
        self.memory_ai[index] = current_action_mask
        self.memory_sa[index] = next_action_mask
        self.memory_history_ai[index] = history_current_mask
        self.memory_history_sa[index] = history_next_mask
        self.memory_done[index, 0] = done_value
        if self.preference_dim:
            self.memory_preference[index] = preference_value

        current_size = min(self.mem_cntr, self.mem_size)
        if current_size > 0:
            max_priority = float(np.max(self.priorities[:current_size]))
        else:
            max_priority = 1.0
        if not np.isfinite(max_priority) or max_priority <= 0.0:
            max_priority = 1.0
        self.priorities[index] = max_priority
        self.mem_cntr += 1

    def beta_by_frame(self):
        progress = min(1.0, self.sample_step / float(self.beta_frames))
        return self.beta_start + progress * (1.0 - self.beta_start)

    def sample_buffer(self, batch_size):
        batch_size = self._validate_positive_int("batch_size", batch_size)
        current_size = len(self)
        if current_size == 0:
            raise ValueError("Replay buffer is empty.")

        priorities = np.asarray(
            self.priorities[:current_size],
            dtype=np.float64,
        )
        priorities = np.where(
            np.isfinite(priorities) & (priorities > 0.0),
            priorities,
            self.eps,
        )

        scaled = np.power(priorities, self.alpha, dtype=np.float64)
        scaled_sum = float(np.sum(scaled))
        if not np.isfinite(scaled_sum) or scaled_sum <= 0.0:
            probabilities = np.full(
                current_size,
                1.0 / current_size,
                dtype=np.float64,
            )
        else:
            probabilities = scaled / scaled_sum

        # The learner already waits until len(buffer) >= batch_size.  Keeping the
        # fallback makes the buffer safe when called independently in tests.
        replace = current_size < batch_size
        sample_indices = np.random.choice(
            current_size,
            size=batch_size,
            replace=replace,
            p=probabilities,
        )

        self.sample_step += 1
        beta = self.beta_by_frame()

        sampled_probabilities = np.maximum(
            probabilities[sample_indices],
            np.finfo(np.float64).tiny,
        )
        importance_weights = np.power(
            current_size * sampled_probabilities,
            -beta,
        )
        max_weight = float(np.max(importance_weights))
        if not np.isfinite(max_weight) or max_weight <= 0.0:
            importance_weights = np.ones_like(importance_weights)
        else:
            importance_weights /= max_weight
        importance_weights = importance_weights.astype(np.float32).reshape(-1, 1)

        batch_s = np.repeat(self.state_s[np.newaxis, :, :], batch_size, axis=0)
        batch_a = self.memory_a[sample_indices].copy()
        batch_r = self.memory_r[sample_indices].copy()
        batch_current_mask = self.memory_ai[sample_indices].copy()
        batch_next_mask = self.memory_sa[sample_indices].copy()
        batch_history_current_mask = self.memory_history_ai[sample_indices].copy()
        batch_history_next_mask = self.memory_history_sa[sample_indices].copy()
        batch_done = self.memory_done[sample_indices].copy()
        batch_preference = self.memory_preference[sample_indices].copy()

        return (
            batch_s,
            batch_a,
            batch_r,
            batch_current_mask,
            batch_next_mask,
            batch_history_current_mask,
            batch_history_next_mask,
            batch_done,
            sample_indices,
            importance_weights,
            batch_preference,
        )

    def update_priorities(self, indices, td_errors):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        errors = np.asarray(td_errors, dtype=np.float64).reshape(-1)
        if indices.size != errors.size:
            raise ValueError(
                "indices and td_errors must have the same length: "
                f"{indices.size} != {errors.size}"
            )

        current_size = len(self)
        if np.any(indices < 0) or np.any(indices >= current_size):
            raise IndexError(
                f"priority index outside current buffer range [0, {current_size - 1}]."
            )
        if not np.isfinite(errors).all():
            raise ValueError("td_errors contain NaN or Inf.")

        new_priorities = np.clip(
            np.abs(errors) + self.eps,
            self.eps,
            5.0,
        ).astype(np.float32)

        # Sampling with replacement can return the same index more than once.
        # Keep the largest TD error for that transition instead of allowing the
        # final duplicate occurrence to overwrite it with a smaller value.
        for index in np.unique(indices):
            self.priorities[index] = np.max(new_priorities[indices == index])

    def clear(self):
        self.state_s.fill(0.0)
        self.memory_a.fill(0)
        self.memory_r.fill(0.0)
        self.memory_ai.fill(0.0)
        self.memory_sa.fill(0.0)
        self.memory_done.fill(0.0)
        self.memory_preference.fill(0.0)
        self.priorities.fill(0.0)
        self.mem_cntr = 0
        self.sample_step = 0

    def __len__(self):
        return min(self.mem_cntr, self.mem_size)
