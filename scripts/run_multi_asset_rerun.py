#!/usr/bin/env python3
"""Rerun all price variants on AAPL, MSFT, SPY, and IWM on one calendar.

IWM.US is explicitly the investable Russell 2000 ETF proxy, not the Russell
2000 index. The script diagnoses the earlier EODHD timeout, retries the stock
universe in bounded batches, and never prints credentials or request URLs with
secrets.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
from typing import Any
import pandas as pd
from ito_quant.market_data import get_daily_ohlcv, get_daily_prices
from simple_backtest import Backtest, BacktestConfig
from simple_backtest.fundamental_signals import DEFAULT_STOCK_UNIVERSE, build_point_in_time_fundamental_signals
from simple_backtest.strategy.itoflow_signals import FullyInvestedBuyAndHoldStrategy, ItoflowSignalStrategy
from simple_backtest.strategy import RSIStrategy
from scripts.run_itoflow_signal_suite import run_strategy, add_deltas, run_fundamental_variants

TARGETS = ["AAPL.US", "MSFT.US", "SPY.US", "IWM.US"]
VOO = "VOO.US"


def ohlcv(symbol, start, end):
    x = get_daily_ohlcv(symbol, start_date=start, end_date=end, fill_method=None, allow_partial_coverage=True, include_raw_close=True)
    if x is None or x.empty: raise RuntimeError(f"No Itoflow OHLCV returned for {symbol}")
    x = x.rename(columns={c: str(c).title() for c in x.columns})
    x.index = pd.to_datetime(x.index, utc=True)
    return x.sort_index().dropna(subset=["Open","High","Low","Close","Volume"])


def batch_prices(symbols, start, end, batch_size=6, retries=2, backoff=1.0):
    outcomes=[]; frames={}
    for i in range(0, len(symbols), batch_size):
        batch=symbols[i:i+batch_size]; last=None; ok=False
        for attempt in range(1,retries+2):
            try:
                frame=get_daily_prices(batch,start_date=start,end_date=end,fill_method=None,allow_partial=True)
                if frame is None: raise RuntimeError("Itoflow returned no price frame")
                for s in batch:
                    if s in frame.columns and frame[s].dropna().size:
                        frames[s]=pd.to_numeric(frame[s],errors="coerce").dropna()
                ok=True; last=None; outcomes.append({"symbols":batch,"attempts":attempt,"status":"success","error":None}); break
            except Exception as exc:
                last=f"{type(exc).__name__}: {exc}"; time.sleep(backoff*(2**(attempt-1)))
        if not ok: outcomes.append({"symbols":batch,"attempts":retries+1,"status":"failed","error":last})
    return frames,outcomes


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",type=Path,default=Path("research_outputs/multi_asset_rerun")); ap.add_argument("--batch-size",type=int,default=6); ap.add_argument("--retries",type=int,default=2); ap.add_argument("--backoff",type=float,default=1.0); args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    discovery={s:ohlcv(s,"2000-01-01",None) for s in TARGETS+[VOO]}
    latest=min(x.index[-1] for x in discovery.values()); five=latest-pd.DateOffset(years=5)
    common=discovery[TARGETS[0]].index
    for s in TARGETS[1:]+[VOO]: common=common.intersection(discovery[s].index)
    eval_idx=common[common>=five]; start_pos=max(0,common.get_loc(eval_idx[0])-300); run_idx=common[start_pos:]
    holdout=eval_idx[0]; end=eval_idx[-1]
    panels={s:discovery[s].loc[run_idx] for s in TARGETS+[VOO]}
    settings={"branch":"research/itoflow-rsi-news-signal","commit":"ad062bf9e6ca699794a44ec71c5bef1f5034493b","latest_close":str(latest.date()),"warmup_start":str(run_idx[0].date()),"holdout_start":str(holdout.date()),"holdout_end":str(end.date()),"common_observations":len(eval_idx),"settings":"$10,000; open execution; 0.1% commission; final liquidation; 300-row warm-up; unchanged parameters","symbols":{"IWM.US":"investable Russell 2000 ETF proxy, not the index","VOO.US":"S&P 500 ETF benchmark"}}
    rows=[]
    for s in TARGETS:
        proxy=panels[VOO]["Close"] if s=="SPY.US" else panels["SPY.US"]["Close"]
        for variant in ["rsi_baseline","itoflow_rsi","mean_reversion","supertrend","dip_score","vol_scaled","combined"]:
            if variant=="rsi_baseline": st=RSIStrategy(period=14,oversold=30,overbought=70,shares=10,name=f"{s}:{variant}")
            else: st=ItoflowSignalStrategy(variant=variant,symbol=s,market_prices=proxy,fixed_shares=10,name=f"{s}:{variant}")
            rows.append(run_strategy(st.get_name(),st,panels[s],str(holdout.date())))
        st=FullyInvestedBuyAndHoldStrategy(name=f"{s}:buy_hold"); rows.append(run_strategy(st.get_name(),st,panels[s],str(holdout.date())))
    voo=FullyInvestedBuyAndHoldStrategy(name="VOO.US:buy_hold"); rows.append(run_strategy(voo.get_name(),voo,panels[VOO],str(holdout.date())))
    results=add_deltas(rows)
    results.to_csv(args.output_dir/"price_comparison.csv",index=False)
    for s,x in panels.items(): x.to_csv(args.output_dir/f"{s.replace('.','_')}_ohlcv.csv")
    # Diagnose dated-universe inputs with bounded batches; no secrets in error text.
    stock_prices,batches=batch_prices(DEFAULT_STOCK_UNIVERSE,str(run_idx[0].date()),str(end.date()),args.batch_size,args.retries,args.backoff)
    fund_rows=[]; fund_status="unavailable"; fund_reason=None
    if set(DEFAULT_STOCK_UNIVERSE)-set(stock_prices): fund_reason=f"Batched Itoflow get_daily_prices incomplete; missing symbols: {sorted(set(DEFAULT_STOCK_UNIVERSE)-set(stock_prices))}"
    else:
        dates=pd.DatetimeIndex(pd.Series(panels["AAPL.US"].index).groupby(panels["AAPL.US"].index.to_period("M")).min().values).tz_localize("UTC")
        try:
            fr=build_point_in_time_fundamental_signals(DEFAULT_STOCK_UNIVERSE,stock_prices,dates,exchange="US"); fund_status=fr.status; fund_reason=fr.unavailable_reason
            for s in ["AAPL.US","MSFT.US"]: fund_rows += run_fundamental_variants(s,panels[s],panels["SPY.US"]["Close"],fr.scores_by_symbol.get(s),str(holdout.date()))
        except Exception as exc: fund_reason=f"Itoflow dated fundamentals failed: {type(exc).__name__}: {exc}"
    if not fund_rows:
        for s in ["AAPL.US","MSFT.US"]:
            for v in ["value_gate","quality_gate","value_quality_gate"]: fund_rows.append({"strategy":f"{s}:{v}","status":"unavailable","unavailable_reason":fund_reason})
    pd.DataFrame(fund_rows).to_csv(args.output_dir/"fundamental_comparison.csv",index=False)
    manifest={"settings":settings,"provider_diagnosis":{"prior_error":"HTTPSConnectionPool(host='eodhd.com', port=443): Read timed out (read timeout about 14.99 seconds)","interpretation":"Network read timeout to host eodhd.com over HTTPS port 443; not an HTTP status code. No authentication or rate-limit conclusion is inferred.","method":"ito_quant.market_data.get_daily_prices","request":"24-stock universe price fetch","batch_size":args.batch_size,"retries":args.retries,"batches":batches,"propagated_failure":"The previous single-call try/except propagated one timeout message to every symbol; this rerun records per-batch outcomes."},"fundamentals":{"status":fund_status,"reason":fund_reason,"ETF_fundamentals":"not applicable for SPY.US or IWM.US"},"news":"Not used; GDELT remains unavailable/rate-limited.","run_status":"price suite completed; fundamental rows are available or explicitly unavailable per above","artifact_paths":["price_comparison.csv","fundamental_comparison.csv","diagnostics.json"]}
    (args.output_dir/"diagnostics.json").write_text(json.dumps(manifest,indent=2,default=str))
    print(results.to_string(index=False)); print(json.dumps(manifest,indent=2,default=str))

if __name__=="__main__": main()
