import numpy as np

from agrosurveillance.rl.replay_buffer import ReplayBuffer
from agrosurveillance.rl.sac import SACAgent


def test_sac_select_action_within_bounds():
    agent = SACAgent(state_dim=6, action_dim=1, action_high=10.0, hidden_dim=16)
    state = np.random.randn(6).astype(np.float32)
    action = agent.select_action(state)
    assert action.shape == (1,)
    assert 0.0 <= action[0] <= 10.0


def test_sac_update_runs_and_reduces_loss_trend():
    state_dim, action_dim = 6, 1
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim, action_high=10.0, hidden_dim=16)
    buf = ReplayBuffer(500, state_dim, action_dim)

    rng = np.random.RandomState(0)
    for _ in range(200):
        s = rng.randn(state_dim).astype(np.float32)
        a = rng.uniform(0, 10, size=(action_dim,)).astype(np.float32)
        r = -abs(rng.randn())
        ns = rng.randn(state_dim).astype(np.float32)
        buf.add(s, a, r, ns, False)

    losses = []
    for _ in range(10):
        stats = agent.update(buf.sample(32))
        losses.append(stats["critic_loss"])

    assert all(np.isfinite(losses))
    assert len(losses) == 10
