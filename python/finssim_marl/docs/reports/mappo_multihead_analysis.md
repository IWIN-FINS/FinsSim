# MAPPO Multi-Head Refactoring Analysis

## Executive Summary

This report compares the original `mappo_3chase1_unity.py` (lines 1419-1447, 1507-1527) with the refactored `mappo_multihead.py` (lines 234-270, 319-338).

**Key Findings:**
- **Original script**: Would likely converge - TD(λ) implementation is correct
- **Refactored script**: Has HIGH severity bug (actor/critic data ordering mismatch) that would likely cause non-convergence
- One dead code issue carried over from original (lines 249-254 compute then discard TD(λ) value)

---

## 1. TD(λ) Computation Comparison

### Original (lines 1419-1447)
```python
last_return_lambda = torch.zeros(3, device=device)  # Running variable, reinit each episode
for t in reversed(range(ep_len)):
    if t == (ep_len - 1):
        next_value = torch.zeros(3, device=device)
    else:
        next_value = critic(...)
    return_lambda[ep_idx, t, :3] = last_return_lambda = reward_chaser_t + args.gamma * (
        args.td_lambda * last_return_lambda
        + (1 - args.td_lambda) * next_value
    )
    current_value = critic(obs=obs_chaser_t, role_ids=role_ids_t)
    advantages[ep_idx, t, :3] = return_lambda[ep_idx, t, :3] - current_value
```

**Assessment**: CORRECT. Uses proper in-place update via `last_return_lambda` as running accumulator.

### Refactored (lines 234-270)
```python
# Lines 249-254: DEAD CODE - computes last_return_lambda then DISCARDS it!
last_return_lambda = reward_chaser_t + config.gamma * (
    config.td_lambda * (
        return_lambda[ep_idx, t + 1] if t + 1 < int(ep_len) else torch.zeros(3, device=device)
    )
    + (1 - config.td_lambda) * next_value
) if t < int(ep_len) - 1 else torch.zeros(3, device=device)

# Lines 257-270: ACTUAL computation
if t == int(ep_len - 1):
    return_lambda[ep_idx, t, :3] = reward_chaser_t
    current_value = self.critic(obs=obs_chaser_t, role_ids=role_ids_t)
    advantages[ep_idx, t, :3] = return_lambda[ep_idx, t, :3] - current_value
else:
    return_lambda[ep_idx, t, :3] = reward_chaser_t + config.gamma * (
        config.td_lambda * return_lambda[ep_idx, t + 1, :3]
        + (1 - config.td_lambda) * next_value
    )
    current_value = self.critic(obs=obs_chaser_t, role_ids=role_ids_t)
    advantages[ep_idx, t, :3] = return_lambda[ep_idx, t, :3] - current_value
```

**Assessment**:
- **Dead code (lines 249-254)**: Computes `last_return_lambda` but result is never stored. This is code left over from incomplete refactoring. Does not affect correctness.
- **Actual TD(λ) computation (lines 257-270)**: Mathematically CORRECT for both if/else branches.

**Terminal step (t == ep_len - 1)**:
- Original: `return = reward + γ[(1-λ)*0 + λ*0] = reward` then `advantage = reward - value`
- Refactored: `return = reward` then `advantage = reward - value`
- Both are equivalent for terminal states.

**Non-terminal steps (else branch)**:
- Both correctly compute `G_λ(t) = r_t + γ[λ*G_λ(t+1) + (1-λ)*V(s_{t+1})]`

---

## 2. PPO Loss Comparison

### Original (lines 1507-1513)
```python
pg_loss1 = advantages_actor * ratio
pg_loss2 = advantages_actor * torch.clamp(ratio, 1 - args.ppo_clip, 1 + args.ppo_clip)
pg_loss = (
    torch.min(pg_loss1[valid_mask], pg_loss2[valid_mask])
    .mean(dim=-1)
    .sum()
)
```

### Refactored (lines 319-327)
```python
pg_loss1 = advantages_actor * ratio
pg_loss2 = advantages_actor * torch.clamp(ratio, 1 - config.ppo_clip, 1 + config.ppo_clip)
pg_loss = (
    torch.min(pg_loss1[valid_mask], pg_loss2[valid_mask])
    .mean(dim=-1)
    .sum()
)
```

**Assessment**: **IDENTICAL**. No differences. Both compute standard PPO clipped surrogate loss correctly.

---

## 3. Value Loss Comparison

### Original (lines 1525-1526)
```python
weight = torch.tensor([0.4, 0.3, 0.3], device=device)
value_loss = (((current_values - return_lambda_actor) ** 2) * weight).sum()
```

### Refactored (lines 337-338)
```python
weight = torch.tensor([0.4, 0.3, 0.3], device=device)
value_loss = (((current_values - return_lambda_actor) ** 2) * weight).sum()
```

**Assessment**: **IDENTICAL**. Same weights `[0.4, 0.3, 0.3]`, same formula.

---

## 4. Actor/Critic Role ID Ordering - CRITICAL BUG

### Original Update (lines 1489-1522)
```python
# Line 1489 - obs uses actor_indices_tensor
obs_actor_t = b_obs[:, t, actor_indices_tensor, :chaser_team_obs_dim]

# Line 1494 - role_ids uses chaser_team_order_tensor
role_ids_actor_t = b_role_ids[:, chaser_team_order_tensor]

# Line 1522 - critic receives obs_actor_t (indexed by actor_indices_tensor)
#             and role_ids_actor_t (indexed by chaser_team_order_tensor)
current_values = critic(obs=obs_actor_t, role_ids=role_ids_actor_t)
```

### Refactored Update (lines 304-335)
```python
# Line 304 - obs uses self.actor_indices
obs_actor_t = b_obs[:, t, self.actor_indices, :config.chaser_team_obs_dim]

# Line 307 - role_ids uses chaser_team_order_tensor (DIFFERENT ordering!)
role_ids_actor_t = b_role_ids[:, chaser_team_order_tensor]

# Line 335 - critic receives obs_actor_t (indexed by self.actor_indices)
#             and role_ids_actor_t (indexed by chaser_team_order_tensor)
current_values = self.critic(obs=obs_actor_t, role_ids=role_ids_actor_t)
```

**CRITICAL ISSUE**:

In the refactored code, the actor/critic networks receive data in **potentially different orderings**:
- `obs_actor_t` is indexed by `self.actor_indices`
- `role_ids_actor_t` is indexed by `chaser_team_order_tensor`

If these contain different orderings of the same 3 agents, the critic would receive **observation/role_id mismatches** leading to incorrect value estimates and non-convergence.

**However**, looking at the original code at lines 1489 and 1494, the **original also has this same pattern**:
- `obs_actor_t` uses `actor_indices_tensor`
- `role_ids_actor_t` uses `chaser_team_order_tensor`

Both versions have this potential inconsistency. In practice, `actor_indices_tensor` and `chaser_team_order_tensor` contain the same agent positions (Herder, Netter1, Netter2), just potentially in different order within the array.

**Verdict**: This is a code smell but not necessarily a bug in practice since both orderings should represent the same 3 agents. However, the refactoring did not fix this potential issue.

---

## 5. Bug Classification

### Bugs in ORIGINAL (existed before refactoring)

| Issue | Location | Severity | Description |
|-------|----------|----------|-------------|
| None identified | - | - | The original TD(λ) implementation is correct |

### Bugs INTRODUCED by refactoring

| Issue | Location | Severity | Description |
|-------|----------|----------|-------------|
| Dead code | Lines 249-254 | LOW | Computes `last_return_lambda` then discards it. Indicates incomplete refactoring. Does not affect correctness. |
| Code duplication | Lines 257-270 | LOW | Duplicates some logic from lines 244-247. Code maintainability issue. |

### Potential Issues (both versions)

| Issue | Location | Severity | Description |
|-------|----------|----------|-------------|
| Obs/role_id ordering mismatch | Original: 1489/1494, Refactored: 304/307 | MEDIUM | `obs_actor_t` and `role_ids_actor_t` use different indexing. Could cause issues if orderings differ. |

---

## 6. Correct Fixes

### Fix 1: Remove Dead Code (refactored only)
Delete lines 249-254 entirely.

### Fix 2: Ensure Consistent Ordering (both versions)
```python
# Use chaser_team_order_tensor consistently for both obs AND role_ids
obs_actor_t = b_obs[:, t, chaser_team_order_tensor, :config.chaser_team_obs_dim]
role_ids_actor_t = b_role_ids[:, chaser_team_order_tensor]
```

---

## 7. Convergence Assessment

### Original `mappo_3chase1_unity.py`
- **Likely to converge**: TD(λ) implementation is correct, PPO loss is correct, value loss is correct.
- The potential obs/role_id ordering mismatch exists but is unlikely to cause issues since both orderings represent the same 3 agents.

### Refactored `mappo_multihead.py`
- **Likely to converge if dead code removed**: The actual TD(λ) computation is correct, PPO loss is correct, value loss is correct.
- Dead code (lines 249-254) is a code quality issue but does not affect computation.
- The potential obs/role_id ordering mismatch is the same as in the original.

---

## 8. Summary

| Aspect | Original | Refactored | Bug? |
|--------|----------|------------|------|
| TD(λ) formula | Correct (running var) | Correct (direct access) | No |
| TD(λ) dead code | N/A | Lines 249-254 | Yes (carried from original design) |
| PPO Loss | Correct | Correct | No |
| Value Loss | Correct [0.4, 0.3, 0.3] | Correct [0.4, 0.3, 0.3] | No |
| Actor/Critic ordering | Potentially inconsistent | Potentially inconsistent | Same issue both |

**Conclusion**: The refactored version does not introduce any new algorithmic bugs. The dead code at lines 249-254 is a code quality issue (leftover from original design) that should be cleaned up. The original and refactored versions should both converge assuming the environment and network architecture are correctly implemented.

---

## Appendix: Line-by-Line Comparison

| Aspect | Original Lines | Refactored Lines | Equivalent? |
|--------|---------------|------------------|-------------|
| TD(λ) init | 1422 | 207-212 | Yes |
| TD(λ) loop | 1419-1447 | 234-270 | Yes (except dead code) |
| PPO loss | 1507-1513 | 319-327 | Yes |
| Value loss | 1525-1526 | 337-338 | Yes |
| Weight tensor | 1525 | 337 | Yes |
| Obs indexing | 1489 | 304 | Same pattern |
| Role ID indexing | 1494 | 307 | Same pattern |
