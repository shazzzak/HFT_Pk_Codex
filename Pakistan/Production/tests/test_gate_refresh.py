"""Gate checks must observe real risk decisions."""
# Supply the replay's small book interface without reading captured data.
from types import SimpleNamespace
# Use the production messages.
from core.model import DesiredQuotes, QuoteIntent, Side
# Exercise the gate's real gateway construction.
from sim import gate


# A deliberately tiny limit must appear in the gate's rejection evidence.
def test_gate_counts_actual_risk_rejections(monkeypatch):
    # Replace only market replay, retaining risk and OMS behavior.
    class Replay:
        # Accept the same construction arguments as EngineReplay.
        def __init__(self, strategy, adapter, oms, symbol, cfg):
            # Keep the real order manager.
            self.oms = oms
            # No published exchange bands are needed for this position test.
            self.book = SimpleNamespace(limit_up=None, limit_dn=None)
            # Match the replay's statistics interface.
            self.engine_stats = {}

        # Request an order which exceeds the declared share limit.
        def run(self, events, snapshots):
            # The price is at the reference, so only position risk should bind.
            self.oms.set_desired(DesiredQuotes("OGDC", bid=QuoteIntent(Side.BUY, 10000, 50)))
            # Route through the production gateway.
            assert self.oms.reconcile(1000, "2026-06-30", {"OGDC": 10000}) == []

    # Install the replay double locally for this test only.
    monkeypatch.setattr(gate, "EngineReplay", Replay)
    # Ask the real gate builder for a ten-share cap.
    result = gate.run_engine([], {}, dict(gate.R.MICRO_PARAMS, session_scale=1.0), 0, 10000, 10000, "OGDC", "2026-06-30", True, "exact", 10)
    # The former disconnected counter would incorrectly report zero.
    assert result.engine_stats["gateway_rejections"] == 1
    # Preserve the exact rejecting control for diagnosis.
    assert result.risk_rejections[0]["check"] == "position_limit"
