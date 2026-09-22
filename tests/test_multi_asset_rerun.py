import pandas as pd
from scripts.run_multi_asset_rerun import target_deltas, add_fundamental_deltas

def test_target_deltas_use_each_target_baseline_and_voo():
    frame = pd.DataFrame([
        {"strategy":"AAPL.US:rsi_baseline","total_return":1.,"sharpe_ratio":.1,"max_drawdown":2.},
        {"strategy":"AAPL.US:combined","total_return":3.,"sharpe_ratio":.3,"max_drawdown":1.},
        {"strategy":"MSFT.US:rsi_baseline","total_return":10.,"sharpe_ratio":1.,"max_drawdown":8.},
        {"strategy":"MSFT.US:combined","total_return":12.,"sharpe_ratio":1.2,"max_drawdown":7.},
        {"strategy":"VOO.US:buy_hold","total_return":20.,"sharpe_ratio":2.,"max_drawdown":10.},
    ])
    out=target_deltas(frame,["AAPL.US","MSFT.US"],"VOO.US:buy_hold")
    assert out.loc[out.strategy=="MSFT.US:rsi_baseline","return_delta_vs_rsi_pct_points"].iloc[0] == 0
    assert out.loc[out.strategy=="MSFT.US:combined","return_delta_vs_rsi_pct_points"].iloc[0] == 2
    assert out.loc[out.strategy=="AAPL.US:combined","return_delta_vs_voo_benchmark_pct_points"].iloc[0] == -17

def test_batch_prices_records_failed_batch_without_propagating(monkeypatch):
    import scripts.run_multi_asset_rerun as mod
    calls=[]
    def fake(symbols, **kwargs):
        calls.append(list(symbols))
        if symbols[0] == "BAD.US": raise RuntimeError("synthetic timeout")
        return pd.DataFrame({symbols[0]: [1., 2.]}, index=pd.date_range("2020-01-01", periods=2))
    monkeypatch.setattr(mod, "get_daily_prices", fake)
    frames, outcomes = mod.batch_prices(["GOOD.US","BAD.US"], "2020-01-01", "2020-01-03", batch_size=1, retries=1, backoff=0)
    assert "GOOD.US" in frames
    assert outcomes[-1]["status"] == "failed"
    assert outcomes[-1]["attempts"] == 2

def test_fundamental_rows_reconcile_against_target_baseline_and_voo():
    import scripts.run_multi_asset_rerun as mod
    rows = pd.DataFrame([
        {"strategy":"AAPL.US:rsi_baseline","total_return":3.0,"sharpe_ratio":.3,"max_drawdown":4.0},
        {"strategy":"AAPL.US:value_gate","total_return":5.0,"sharpe_ratio":.5,"max_drawdown":2.0},
        {"strategy":"MSFT.US:rsi_baseline","total_return":4.0,"sharpe_ratio":.4,"max_drawdown":8.0},
        {"strategy":"MSFT.US:quality_gate","total_return":6.0,"sharpe_ratio":.6,"max_drawdown":7.0},
        {"strategy":"VOO.US:buy_hold","total_return":10.0,"sharpe_ratio":1.0,"max_drawdown":10.0},
    ])
    fundamentals=rows.iloc[[1,3]].copy()
    out=mod.add_fundamental_deltas(fundamentals,rows,["AAPL.US","MSFT.US"],"VOO.US:buy_hold")
    a=out.loc[out.strategy=="AAPL.US:value_gate"].iloc[0]
    m=out.loc[out.strategy=="MSFT.US:quality_gate"].iloc[0]
    assert (a.return_delta_vs_rsi_pct_points, a.sharpe_delta_vs_rsi, a.drawdown_delta_vs_rsi_pct_points)==(2.0,.2,-2.0)
    assert (m.return_delta_vs_voo_benchmark_pct_points, m.sharpe_delta_vs_voo_benchmark, m.drawdown_delta_vs_voo_benchmark_pct_points)==(-4.0,-.4,-3.0)
    artifact=pd.read_csv("research_outputs/multi_asset_rerun/fundamental_comparison.csv")
    required={"return_delta_vs_rsi_pct_points","sharpe_delta_vs_rsi","drawdown_delta_vs_rsi_pct_points","return_delta_vs_voo_benchmark_pct_points","sharpe_delta_vs_voo_benchmark","drawdown_delta_vs_voo_benchmark_pct_points"}
    assert required.issubset(artifact.columns)
