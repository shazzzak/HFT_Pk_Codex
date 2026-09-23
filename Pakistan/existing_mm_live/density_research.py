"""Isolated PSX density study: build guarded features, walk forward, export evidence."""
# Parse an explicit research scope instead of silently rebuilding the canonical store.
import argparse
# Freeze code/data bytes before dispatch and verify them after analysis.
import hashlib
# Preserve configuration, failures and model transforms as readable artifacts.
import json
# Report local timestamps in the user's requested heartbeat format.
from datetime import datetime
# Resolve source versus external result destinations.
from pathlib import Path
# Bound concurrency and keep only the parent process printing progress.
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
# Spawn clean workers and publish throttled status through a shared manager.
import multiprocessing
# Control external plotting caches and record runtime provenance.
import os
# Capture Python version and source module paths.
import sys
# Measure durations and throttle global progress output.
import time
# Run exactly one parent-side timer without touching market/model state.
import threading
# Catch stack traces in cell evidence without hiding input failures.
import traceback
# Reuse the canonical data/config paths, not historical hardcoded roots.
import config_pk as C
# Use the same parsed-store schema, REG selection and event construction as the backtest.
import run_legacy_mm as R
# Build compact numerical feature frames and CSV research outputs.
import pandas as pd
# Use finite checks in research coverage and summary formatting.
import numpy as np
# Reuse the dedicated known-price geometry collector.
from density_collector import collect
# Keep analysis separate from any strategy or live engine code.
from density_validation import evaluate_symbol, liquidity_labels, summarize, plot_results
# Cover exactly the declared depth windows in quality reports.
from book_density import DEPTHS


# Serialize complete JSON documents with nonfinite fields rejected by default.
def save(path,value):
    # Scientific NaN values belong in parquet/CSV; manifests must remain strict JSON.
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False,default=str)+"\n")


# Stream bytes to avoid loading large parquet inputs into memory just to hash them.
def digest(path):
    # Preserve standard SHA-256 provenance semantics.
    value = hashlib.sha256()
    # Hash exact bytes in bounded read blocks.
    with Path(path).open("rb") as stream:
        # Stop only at a true end-of-file.
        for block in iter(lambda: stream.read(1024*1024),b""):
            # Every consumed byte contributes to the frozen input fingerprint.
            value.update(block)
    # Return a stable textual manifest value.
    return value.hexdigest()


# Own the single console progress stream for both parallel extraction and serial analysis.
class Reporter:
    # Workers publish data, but only this parent prints every fifteen seconds.
    def __init__(self):
        # Keep all duration measurements on one monotonic clock.
        self.started,self.last = time.monotonic(),0.0
        # Track each phase separately so analysis ETA excludes earlier extraction time.
        self.phase_key,self.phase_started = None,self.started
        # Rotate active symbols so concurrent workers do not flood the terminal.
        self.rotation = 0
        # Copy only diagnostic state across the parent timer boundary.
        self.lock,self.stop = threading.Lock(),threading.Event()
        # Begin with an explicit startup phase before the first expensive read.
        self.snapshot = ({},0,0)
        # A daemon reporter cannot hold a failed research process open.
        self.thread = threading.Thread(target=self._loop,daemon=True)
        # Only the parent constructs this object, so workers never print heartbeats.
        self.thread.start()
    # Publish progress while the timer alone owns ordinary console output.
    def tick(self,states,done,total,force=False):
        # Detach lightweight dictionaries from multiprocessing proxy objects.
        with self.lock:
            # No book, model or financial objects cross into the timer thread.
            self.snapshot = ({key:dict(value) for key,value in states.items()},done,total)
        # Explicit completion/error announcements may bypass the periodic throttle.
        if force:
            # Still emit only one compact status line.
            self._emit(states,done,total,True)
    # Emit a status snapshot at most once per fixed fifteen-second timer interval.
    def _loop(self):
        # Event.wait permits immediate shutdown without waiting another interval.
        while not self.stop.wait(15):
            # Copy a consistent diagnostic snapshot before formatting it.
            with self.lock:
                # The writer never holds this lock during slow console output.
                states,done,total = self.snapshot
            # Report liveness even when loading, hashing or fitting is blocked.
            self._emit(states,done,total,True)
    # Stop reporting before the one final completion line.
    def close(self):
        # Wake the timer promptly on ordinary completion.
        self.stop.set()
        # Avoid progress output racing the final verdict.
        self.thread.join()
    # Format one timer-owned progress line without reading live model objects.
    def _emit(self,states,done,total,force=False):
        # Read elapsed time before considering a terminal write.
        now = time.monotonic()
        # Completion/errors may be immediate; ordinary stages share one global throttle.
        if not force and now-self.last < 15:
            # Reading worker progress should not itself create console traffic.
            return
        # Prefer currently active jobs; zero active jobs means a parent-side stage.
        active = list(states.values())
        # Use a transparent waiting state while workers import libraries or finish.
        item = active[self.rotation%len(active)] if active else dict(symbol="-",date="-",stage="waiting",done=0,total=0)
        # Advance the round-robin selection once per emitted line only.
        self.rotation += 1
        # Keep processed-event percentage distinct from completed symbol-days.
        fraction = item["done"]/item["total"] if item["total"] else None
        # Name the exact stage whose fraction is being shown.
        percent = f"{100*fraction:.0f}%" if fraction is not None else "pending"
        # A run ETA uses completed comparable cells only; pre-completion estimates are unknown.
        elapsed = now-self.started
        # Do not extrapolate one worker's event fraction to the whole portfolio.
        # Group all worker stages under extraction; keep serial analysis as its own phase.
        phase_key = "Extraction" if item.get("cell_phase",False) else item["stage"]
        # Reset the phase stopwatch at the first report of a new phase.
        if phase_key != self.phase_key:
            # Retain total elapsed separately for the existing user-facing clock.
            self.phase_key,self.phase_started = phase_key,now
        # Walk-forward progress includes completed symbols plus the current date fraction.
        progress = done+(fraction or 0.) if phase_key=="Walk-forward" else done
        # Parent hashing stages report their own file denominator instead of symbol counts.
        target = total
        # Use parent-stage counters when no portfolio denominator exists.
        if not total and item["total"]:
            # This estimate covers only the named stage, not unknown later work.
            progress,target = item["done"],item["total"]
        # Estimate remaining comparable work from elapsed time in this phase only.
        phase_elapsed = now-self.phase_started
        # No truthful rate is available at a phase's very first report.
        eta = f"~{phase_elapsed/progress*(target-progress)/60:.0f}m" if progress>0 and target>progress and phase_elapsed>0 else "estimating"
        # A completed nonempty phase has no remaining work in that phase.
        if target and progress>=target:
            # Subsequent phases receive their own estimates when they begin.
            eta = "0m"
        # Finishing the entire requested phase means no work remains in that phase.
        if total and done == total:
            # This can describe phase completion, not necessarily completion of later analysis.
            eta = "0m"
        # Distinguish this research feature study from an actual trading strategy run.
        print(f"[{datetime.now():%H:%M:%S}] Ticker: {item['symbol']}; Date: {item['date']}; Strategy: density research | Elapsed {elapsed/60:.1f}m | Phase ETA {eta} | Done {done}/{total} | {item['stage']} {percent}",flush=True)
        # Share the same rate limit across every stage and worker.
        self.last = now


# Freeze all inputs while retaining a single concise parent heartbeat.
def freeze(paths,reporter,phase):
    # Keep traversal order deterministic and deduplicate shared source/data inputs.
    paths = sorted(set(Path(path) for path in paths))
    # Store exact before/after hashes in the same key format.
    result = {}
    # Track file count while byte hashing remains an ordinary read-only operation.
    for index,path in enumerate(paths):
        # Show current file progress without printing a separate line for every file.
        reporter.tick({"parent":dict(symbol="-",date="-",stage=phase,done=index,total=len(paths))},0,0)
        # Preserve the existing bounded digest implementation.
        result[str(path)] = digest(path)
    # Return the frozen provenance closure.
    return result


# Build one symbol-day in an isolated process and publish status without terminal writes.
def build_cell(job,folder,settings,status):
    # Use a stable progress key for concurrent symbol/date jobs.
    key = job["symbol"]+":"+job["date"]
    # Throttle interprocess status updates independently of the console's fifteen seconds.
    last = [0.0]
    # Publish lightweight immutable progress data, never a book or database object.
    def post(stage,done=0,total=0,force=False):
        # Avoid serializing manager updates on every market event.
        now = time.monotonic()
        # Stage changes are immediate; regular counters update at most once per second.
        if force or now-last[0] >= 1:
            # Only the parent will format or print this status.
            status[key] = dict(symbol=job["symbol"],date=job["date"],stage=stage,done=done,total=total,cell_phase=True)
            # Record the last actual interprocess update time.
            last[0] = now
    # Every planned cell returns coverage, including unavailable inputs and failures.
    result = dict(job,status="error")
    # Preserve errors on disk while allowing other independently planned cells to finish.
    try:
        # Dataset opening and normalization can be slower than the feature arithmetic.
        post("loading",force=True)
        # Require the intended canonical input root in every fresh worker process.
        if R.PARSED_ROOT.resolve() != C.PARSED_ROOT.resolve():
            # Wrong-store runs must fail rather than produce clean-looking empty output.
            raise ValueError("parsed root mismatch")
        # Open one date's existing parsed tables, read-only.
        datasets = R.open_datasets(job["date"])
        # Missing whole partitions are data errors rather than nontrading instruments.
        if datasets is None:
            # Preserve the requested cell in the failed denominator.
            raise ValueError("missing parsed date partitions")
        # Explicitly project market so read_symbol applies its REG predicate; missing market fails closed.
        updates = R.read_symbol(datasets["ob_updates"],list(dict.fromkeys([*R.REQ_UPDATES,"market"])),job["symbol"],market="REG")
        # Read snapshots with the same market filter as the reconciled baseline.
        snapshots = R.read_symbol(datasets["ob_snapshot"],R.REQ_SNAP,job["symbol"],market="REG")
        # Project market on trades too, so non-REG executions cannot update this book.
        trades = R.read_symbol(datasets["trades"],list(dict.fromkeys([*R.REQ_TRADES,"market"])),job["symbol"],market="REG")
        # Historical turnover is a liquidity classifier for later days, never today's label input.
        directional = trades[trades.initiator != "AUCTION"] if not trades.empty else trades
        # Record actual observed turnover even if no valid geometric samples exist.
        result["turnover"] = float((directional.price*directional.qty).sum()) if not directional.empty else 0.0
        # No snapshot/trade coverage is an explicit unavailable research cell.
        if snapshots.empty or trades.empty:
            # Do not silently call missing observations zero predictive power.
            result.update(status="unavailable",reason="no snapshots or trades",sampled_rows=0)
        # Valid raw inputs can be replayed without loading calibration or strategy state.
        else:
            # Preserve event ordering and all historical book reconstruction semantics.
            events,groups,_ = R.build_events(updates,snapshots,trades)
            # Audit source clock granularity without assuming whole-second values are precise.
            result["timing"] = {}
            # Keep snapshot, update and trade diagnostics separate.
            for kind in ("S","U","T"):
                # Inspect both clocks on the exact decoded records used below.
                selected = [event for event in events if event[3]==kind]
                # Missing event classes are explicit rather than assigned fabricated timing.
                if selected:
                    # Capture time is actual local observation availability under this parser.
                    exchange = np.array([int(event[0]) for event in selected],dtype=np.int64)
                    # The source loader converts capture timestamps to integer milliseconds.
                    capture = np.array([int(event[4].ts_cap) for event in selected],dtype=np.int64)
                    # NaT/negative timestamps cannot support causal short-horizon sampling.
                    if np.any(exchange<0) or np.any(capture<0):
                        # Refuse malformed source clocks rather than sorting them into the distant past.
                        raise ValueError("invalid source timestamp")
                    # Preserve precision symptoms, duplicates and clock-offset distribution.
                    result["timing"][kind] = dict(events=len(selected),exchange_whole_second_fraction=float(np.mean(exchange%1000==0)),capture_whole_second_fraction=float(np.mean(capture%1000==0)),capture_unique_ms_fraction=float(len(np.unique(capture))/len(capture)),capture_minus_exchange_ms_p50=float(np.median(capture-exchange)),negative_clock_difference_fraction=float(np.mean(capture<exchange)))
            # Prefer observation-time ordering for subsecond predictive research.
            if settings["clock"] == "capture":
                # Preserve all event payloads while replacing only the replay timeline key.
                events = [(int(event[4].ts_cap),*event[1:]) for event in events]
                # At equal milliseconds keep the established kind/sequence convention.
                events.sort(key=lambda event:(event[0],event[1],event[2]))
            # Announce the true event denominator before the collector starts.
            post("Replay",0,len(events),True)
            # Extract separate versioned features; no orders are ever submitted or simulated.
            frame,coverage = collect(events,groups,settings["tick_size"],settings["grid_ms"],settings["max_age_ms"],settings["horizons"],lambda done,total: post("Replay",done,total),pd_decay_k=settings.get("pd_decay_k",0.1),walk_clip_shares=job.get("clip"),dynamic_window_ms=settings.get("dynamic_window_ms",1000))
            # Keep source-selection loss explicit for thin or halted instruments.
            result.update(coverage)
            # Identify the actual timeline used for both predictors and future markouts.
            result["clock"] = settings["clock"]
            # An empty usable source sample is a coverage result, not a fitted model.
            if frame.empty:
                # Retain turnover metadata for prior-only liquidity classification.
                result.update(status="unavailable",reason="no fresh continuous two-sided grid samples")
            # Persist each complete frame before marking its cell successful.
            else:
                # Tag the exact instrument and date so labels cannot mix sessions later.
                frame["symbol"],frame["date"] = job["symbol"],job["date"]
                # Keep versioned feature data in the explicitly selected external store.
                destination = Path(settings.get("feature_dir",Path(folder)/"features"))/job["symbol"]
                # Workers own disjoint symbol/day files even when directories are shared.
                destination.mkdir(parents=True,exist_ok=True)
                # Encode one immutable feature partition per symbol-day.
                path = destination/f"date={job['date']}.parquet"
                # Write to a temporary name so an interruption cannot look like a complete file.
                temporary = path.with_suffix(".partial")
                # Parquet preserves NaN depth/target exclusions and numeric types.
                frame.to_parquet(temporary,index=False)
                # Publish the completed file atomically on the same filesystem.
                temporary.replace(path)
                # Freeze both content and relative location for later analysis verification.
                result.update(status="ok",feature_file=str(path.resolve()),feature_sha256=digest(path))
                # Record coverage at every depth/horizon, including missing target fractions.
                result["coverage"] = []
                # Report fixed-depth shortfalls separately from short-horizon label availability.
                for depth in DEPTHS:
                    # Match the module's stable column suffix convention.
                    tag = str(depth) if depth is not None else "all"
                    # Label freshness can vary by horizon even on the same source samples.
                    for horizon in settings["horizons"]:
                        # Count actual jointly usable rows rather than imputing unavailable values.
                        result["coverage"].append(dict(depth=tag,horizon_ms=horizon,source_rows=len(frame),valid_depth=int(frame[f"valid_{tag}"].sum()),valid_label=int(frame[f"markout_{horizon}ms_bps"].notna().sum()),paired_rows=int((frame[f"valid_{tag}"] & frame[f"markout_{horizon}ms_bps"].notna()).sum())))
                # Summarize new static and dynamic feature coverage without suppressing incomplete walks.
                result["liquidity_quality"] = dict(rows=len(frame),dynamic_full_rows=int(frame.dyn_full_window.sum()))
                # Full-size prices exist only where the displayed known-price book covers the order.
                if "walk_buy_1x_complete" in frame:
                    # Keep each direction and size's complete count in cell evidence.
                    result["liquidity_quality"].update({f"{side}_{multiple}x_complete":int(frame[f"walk_{side}_{multiple}x_complete"].sum()) for side in ("buy","sell") for multiple in (1,3,5)})
                # All-known-price depth may equal top ten; preserve that scope limitation.
                result["all_equals_top10_fraction"] = float(((frame.visible_bid_levels<=10)&(frame.visible_ask_levels<=10)).mean())
    # Every exception is retained verbatim rather than becoming an implicit skip.
    except Exception:
        # Keep complete diagnostic traceback in the external cell metadata.
        result.update(status="error",error=traceback.format_exc())
    # Persist metadata separately from partially written feature files.
    save(Path(folder)/"cells"/(key.replace(":","_")+".json"),result)
    # Mark the worker finished; the parent removes its progress entry after receipt.
    post("complete",1,1,True)
    # Return small metadata only, avoiding full-frame interprocess transfer.
    return result


# Analyse one instrument at a time to keep portfolio-scale memory bounded.
def analyse(folder,metadata,settings,reporter):
    # Use prior-only turnover tiers within this exact declared universe.
    liquidity = liquidity_labels(metadata,settings["train_window"],settings["min_train_days"])
    # Keep daily model summaries rather than all portfolio feature rows in RAM.
    daily_frames,decile_frames = [],[]
    # Coverage/error states remain explicit in metadata; only complete files are model inputs.
    good = [item for item in metadata if item["status"] == "ok"]
    # A missing universe cannot be turned into a success with empty CSVs.
    if not good:
        # The caller must inspect coverage rather than assume no effect.
        raise ValueError("no usable feature partitions")
    # Model each symbol independently before equal-name/day aggregation.
    symbols = sorted({item["symbol"] for item in good})
    # Preserve empty-fit symbols in an explicit coverage ledger.
    fitted_symbols = []
    # Every symbol retains its own frozen training and test transforms.
    for index,symbol in enumerate(symbols):
        # Load only this symbol's completed chronological partitions.
        files = [item for item in good if item["symbol"]==symbol]
        # Refuse a edited/generated partition rather than silently training on changed data.
        for item in files:
            # Compare bytes produced by the extractor with bytes about to be fitted.
            if digest(Path(folder)/item["feature_file"]) != item["feature_sha256"]:
                # The feature store is part of the evidence chain too.
                raise ValueError("feature partition changed before analysis")
        # Keep instrument sessions together while preserving their date tags.
        frame = pd.concat([pd.read_parquet(Path(folder)/item["feature_file"]) for item in files],ignore_index=True)
        # The parent remains the sole progress printer during serial model fitting.
        def progress(name,date,done,total):
            # Show held-out-date progress rather than implying strategy execution.
            reporter.tick({"model":dict(symbol=name,date=date,stage="Walk-forward",done=done,total=total)},index,len(symbols))
        # Fit every coefficient, scale and clipping threshold on earlier dates only.
        daily,deciles,fits = evaluate_symbol(frame,settings["horizons"],settings["min_train_days"],settings["train_window"],liquidity,progress)
        # Report symbols lacking enough dates/depth instead of dropping them without trace.
        fitted_symbols.append(dict(symbol=symbol,partitions=len(files),fits=len(fits),daily_metrics=len(daily)))
        # Freeze exact coefficient and feature-transform provenance for each held-out date.
        save(Path(folder)/"analysis"/f"fits_{symbol}.json",fits)
        # A missing fitted model is recorded, not represented as a zero-effect row.
        if not daily.empty:
            # Persist per-symbol metrics before proceeding to the next instrument.
            daily.to_parquet(Path(folder)/"analysis"/f"daily_{symbol}.parquet",index=False)
            # Retain compact conditional loss summaries for pooled inference.
            daily_frames.append(daily)
            # Retain training-binned response curves for plotting.
            decile_frames.append(deciles)
    # Model coverage is distinct from raw feature coverage.
    save(Path(folder)/"analysis"/"model_coverage.json",fitted_symbols)
    # No out-of-sample evidence is an explicit incomplete validation result.
    if not daily_frames:
        # Never report success merely because extraction completed.
        raise ValueError("no eligible held-out fits; inspect coverage or expand dates")
    # Combine compact date-level summaries, not all market-event rows.
    daily,deciles = pd.concat(daily_frames,ignore_index=True),pd.concat(decile_frames,ignore_index=True)
    # Report that date-block inference may itself take time on a large universe.
    reporter.tick({"stats":dict(symbol="ALL",date="-",stage="Date-block inference",done=0,total=0)},0,0)
    # Equal-name/date effects and approximate block intervals avoid iid-row inference.
    summary = summarize(daily)
    # Export all hypothesis families, including poor and negative outcomes.
    summary.to_csv(Path(folder)/"analysis"/"density_summary.csv",index=False)
    # Preserve row counts and all held-out daily losses for independent reanalysis.
    daily.to_parquet(Path(folder)/"analysis"/"daily_metrics.parquet",index=False)
    # Preserve actual bin occupancy and tied-quantile behavior.
    deciles.to_parquet(Path(folder)/"analysis"/"heldout_deciles.parquet",index=False)
    # Produce standalone comparison/heat-map/response figures as requested.
    plot_results(summary,deciles,Path(folder)/"analysis")
    # Completion is not a claim of alpha; return descriptive coverage only.
    return dict(fitted_symbols=sum(item["fits"]>0 for item in fitted_symbols),hypotheses=len(summary),heldout_dates=int(daily.date.nunique()),inference_eligible=int(summary.p_approx.notna().sum()))


# Validate frozen production sizing independently of historical data acquisition.
def attach_production_clips(clip_data,jobs):
    # Build a validated map before any output directory or worker is created.
    clip_map = {}
    # Reject duplicate or malformed sizing records instead of selecting one arbitrarily.
    for entry in clip_data["jobs"]:
        # Keep dates and symbols explicit in the sizing identity.
        identity = (str(entry["symbol"]),str(entry["date"]))
        # A production share clip must be a finite positive integer.
        value = entry["clip"]
        # Boolean values and duplicates are not valid production sizing.
        if identity in clip_map or isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0 or int(value)!=value:
            # Refuse ambiguous or malformed manifest data before replay.
            raise ValueError("invalid/duplicate production clip: "+str(identity))
        # Preserve the exact approved research sizing quantity.
        clip_map[identity] = int(value)
    # Every requested symbol-day requires explicit sizing coverage.
    missing = [(job["symbol"],job["date"]) for job in jobs if (job["symbol"],job["date"]) not in clip_map]
    # Missing sizes cannot silently use another symbol's or day's clip.
    if missing:
        # Bound error output while exposing the absent coverage.
        raise ValueError(f"clip manifest missing {len(missing)} selected jobs; examples: {missing[:5]}")
    # Attach the frozen production quantity to each worker's immutable input.
    for job in jobs:
        # Ignore unrelated gate configuration fields in this research job.
        job["clip"] = clip_map[(job["symbol"],job["date"])]
    # Return the same explicit job list with validated clip values attached.
    return jobs


# Freeze the intended experiment and give large work to the user's terminal.
def main():
    # Explicit arguments make repeated runs comparable and auditable.
    parser = argparse.ArgumentParser(description=__doc__)
    # The smoke is a representative named subset, not an assertion that any name is illiquid.
    parser.add_argument("--smoke",action="store_true")
    # An explicit list overrides the default assignment-file universe.
    parser.add_argument("--symbols")
    # Use existing assignment identifiers only to choose the declared research universe.
    parser.add_argument("--assignment",type=Path,default=C.RESULTS_ROOT/"config_assignment_20260915_0043.csv")
    # Read only symbol/date/clip from a frozen production gate manifest.
    parser.add_argument("--clip-manifest",type=Path,required=True)
    # Freeze the causal event-volume lookback independently of label horizons.
    parser.add_argument("--dynamic-window-ms",type=int,default=1000)
    # Twenty days provides a pipeline pilot; longer held-out histories are required for inference.
    parser.add_argument("--days",type=int,default=20)
    # End-date pinning prevents later data arrivals from silently changing a rerun's scope.
    parser.add_argument("--end-date")
    # Capture-clock replay avoids assigning subsecond precision to coarse snapshot exchange times.
    parser.add_argument("--clock",choices=("capture","exchange"),default="capture")
    # User-operated extraction can use multiple isolated workers with one parent heartbeat.
    parser.add_argument("--workers",type=int,default=2)
    # Tick size is explicit, checked against every sampled price and saved in the manifest.
    parser.add_argument("--tick-size",type=float,default=.01)
    # Apply exponential decay per tick from each side's own touch; zero recovers raw OBI.
    parser.add_argument("--pd-decay-k",type=float,default=0.1)
    # Use clock-time samples so busy names do not automatically supply more time weight.
    parser.add_argument("--grid-ms",type=int,default=1000)
    # Stale-source and stale-target exclusions are a declared research sensitivity parameter.
    parser.add_argument("--max-age-ms",type=int,default=2000)
    # Immediate through five-second horizons are fixed before fitting any data.
    parser.add_argument("--horizons-ms",default="100,250,500,1000,5000")
    # Require enough prior sessions for a model before holding out a whole next date.
    parser.add_argument("--min-train-days",type=int,default=10)
    # A bounded rolling window permits reproducible date-by-date coefficient refreshes.
    parser.add_argument("--train-window",type=int,default=20)
    # Separate versioned feature parquet from analysis reports when requested.
    parser.add_argument("--feature-dir",type=Path)
    # All new data, figures and manifests must live outside the source repository.
    parser.add_argument("--output-dir",type=Path,required=True)
    # Parse one readable, multiline terminal command.
    args = parser.parse_args()
    # Reject insufficient or nonsensical scope before any expensive input loading.
    if args.days <= args.min_train_days or args.min_train_days < 10 or args.train_window < args.min_train_days or args.workers < 1 or args.grid_ms <= 0 or args.max_age_ms <= 0 or not np.isfinite(args.tick_size) or args.tick_size <= 0:
        # State the required historical separation clearly.
        parser.error("require days > min-train-days >= 10, train-window >= min-train-days and positive worker/clock/tick settings")
    # Decay must be finite and nonnegative to define a proximity discount.
    if not np.isfinite(args.pd_decay_k) or args.pd_decay_k < 0:
        # Never silently accept a malformed experiment setting.
        parser.error("pd-decay-k must be finite and nonnegative")
    # Dynamic windows must use a positive explicit wall-clock duration.
    if args.dynamic_window_ms<=0:
        # Invalid windows must fail before expensive data work.
        parser.error("positive dynamic-window-ms required")
    # Parse a unique ordered horizon grid in real milliseconds.
    horizons = sorted(set(int(value) for value in args.horizons_ms.split(",")))
    # Nonpositive horizons are contemporaneous rather than future markouts.
    if not horizons or min(horizons)<=0:
        # Reject malformed horizon specifications before building a feature store.
        parser.error("positive markout horizons required")
    # All output must remain separate from the production/code checkout.
    if args.output_dir.resolve().is_relative_to(C.PROJECT_ROOT.resolve()):
        # Prevent accidentally adding gigabytes of generated artifacts to Git.
        parser.error("output-dir must be outside the repository")
    # Resolve an optional dedicated feature-store run directory.
    feature_dir = (args.feature_dir or args.output_dir/"features").resolve()
    # Keep datasets outside source and refuse existing or overlapping output roots.
    if feature_dir.is_relative_to(C.PROJECT_ROOT.resolve()) or feature_dir.exists() or args.output_dir.resolve().is_relative_to(feature_dir):
        # A new run must never replace existing feature tables or report folders.
        parser.error("feature-dir must be a fresh external directory, not an ancestor of output-dir")
    # Refuse a wrong checkout rather than accidentally invoking old project code.
    if Path(__file__).resolve().parent != C.PAKISTAN_ROOT/"existing_mm_live":
        # Staged code is tested synthetically; real research runs use installed source.
        parser.error("install this package into HFT_Pk_Codex/Pakistan/existing_mm_live first")
    # Resolve the explicit research universe without assuming low-priced names are illiquid.
    symbols = args.symbols.split(",") if args.symbols else ["PACE","KEL","PPL","UBL","NRL","MLCF"] if args.smoke else pd.read_csv(args.assignment).symbol.tolist()
    # Deduplicate while preserving the user's requested instrument order.
    symbols = list(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols))
    # Instrument identifiers must not become filesystem traversal components.
    if not symbols or any(not symbol or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for char in symbol) or ".." in symbol for symbol in symbols):
        # Fail before using any identifier in a partition filename.
        parser.error("invalid symbol identifier")
    # Freeze calendar selection once, before any parallel workers open data.
    available = [str(date) for date in R.discover_dates() if not args.end_date or str(date)<=args.end_date]
    # A short store cannot silently shorten the requested sample.
    if len(available)<args.days:
        # Make incomplete historical scope actionable before launching jobs.
        parser.error("insufficient dates in parsed store")
    # Use the most recent declared calendar window, or the pinned end-date window.
    dates = available[-args.days:]
    # Preserve every planned cell, including unavailable symbol-days.
    jobs = [dict(symbol=symbol,date=date) for date in dates for symbol in symbols]
    # Freeze manifest bytes at selection time so later edits cannot silently change sizes.
    clip_hash = digest(args.clip_manifest)
    # Interpret only data fields, never executable parameters from external manifests.
    clip_data = json.loads(args.clip_manifest.read_text())
    # Validate every selected job before creating outputs or reading market rows.
    try:
        # Attach only the explicit production clips from the frozen data manifest.
        attach_production_clips(clip_data,jobs)
    # Convert malformed sizing into an actionable command-line error.
    except ValueError as error:
        # Never fall back to a guessed or uniform clip.
        parser.error(str(error))
    # Record every parameter affecting extraction, labels, models or classification.
    settings = dict(clip_manifest=str(args.clip_manifest.resolve()),clip_manifest_sha256=clip_hash,dynamic_window_ms=args.dynamic_window_ms,pd_decay_k=args.pd_decay_k,feature_dir=str(feature_dir),clock=args.clock,tick_size=args.tick_size,grid_ms=args.grid_ms,max_age_ms=args.max_age_ms,horizons=horizons,min_train_days=args.min_train_days,train_window=args.train_window,symbols=symbols,dates=dates)
    # Never overwrite a prior experiment or an interrupted run's evidence.
    args.output_dir.mkdir(parents=True,exist_ok=False)
    # Create a fresh versioned dataset directory without replacing existing data.
    feature_dir.mkdir(parents=True,exist_ok=False)
    # Keep cell and analysis files in distinct external subdirectories.
    for child in ("cells","analysis","matplotlib_cache"):
        # Create parent-owned output directories before workers begin.
        (args.output_dir/child).mkdir()
    # Ensure matplotlib's first-run font cache stays outside the repository too.
    os.environ["MPLCONFIGDIR"] = str(args.output_dir/"matplotlib_cache")
    # One global reporter enforces the user's concise fifteen-second output policy.
    reporter = Reporter()
    # Freeze the local research tree, including lazily imported snapshot preparation.
    code_root = Path(__file__).resolve().parent
    # Exclude generated/documentation directories while including all executable modules.
    source_paths = {path for path in code_root.rglob("*.py") if not {"docs","__pycache__",".git",".codex",".agents"}.intersection(path.relative_to(code_root).parts)}
    # Preserve source inventory separately so newly added modules are detected at the end.
    paths = set(source_paths)
    # Sizing is a consumed research input with the same integrity requirements as data.
    paths.add(args.clip_manifest.resolve())
    # Assignment bytes matter only when they actually selected the research universe.
    if not args.symbols and not args.smoke:
        # A later assignment change must not silently alter the frozen universe selection.
        paths.add(args.assignment.resolve())
    # Freeze all consumed raw partition files once per date rather than once per symbol.
    data_paths = {path for date in dates for table in ("trades","ob_updates","ob_snapshot") for path in (C.PARSED_ROOT/table/f"date={date}").rglob("*.parquet")}
    # Raw inputs are part of the causal evidence chain, not just their decoded row counts.
    paths.update(data_paths)
    # Initial hashing is read-only and exposes its stage through the global reporter.
    hashes = freeze(paths,reporter,"Input hashing")
    # A changed sizing file invalidates selection before any workers start.
    if hashes[str(args.clip_manifest.resolve())] != clip_hash:
        # Never proceed with clips that do not match the frozen bytes.
        raise ValueError("clip manifest changed during setup")
    # Persist the experiment before any future labels are computed.
    save(args.output_dir/"manifest.json",dict(settings=settings,jobs=jobs,hashes=hashes,python=sys.version,numpy=np.__version__,pandas=pd.__version__,parsed_root=str(C.PARSED_ROOT),clock=args.clock,clock_limit="capture time is observation availability, not proof of exchange-time precision or executable alpha",all_depth="all reconstructed known-price levels; unpriced deep aggregates excluded",live_approved=False))
    # Collect metadata from every planned job, not just successful feature partitions.
    results = []
    # A shared status map contains small diagnostics only, never market/account state.
    with multiprocessing.get_context("spawn").Manager() as manager:
        # The sole console reporter reads one snapshot of worker status at a time.
        status = manager.dict()
        # Spawn clean processes with bounded instrument/day concurrency.
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            # Submit each explicit symbol-day exactly once.
            pending = {pool.submit(build_cell,job,str(args.output_dir),settings,status):job for job in jobs}
            # Poll completion frequently without printing more than once per fifteen seconds.
            while pending:
                # Short waits keep interrupts responsive while the shared throttle controls output.
                ready,_ = wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                # Display one active worker, rotating through the live set.
                reporter.tick(dict(status),len(results),len(jobs))
                # Record every complete worker outcome before submitting further analysis.
                for future in ready:
                    # Preserve planned identity even for an unexpected process-level failure.
                    job = pending.pop(future)
                    # Convert worker crashes into explicit failed cells instead of dropping them.
                    try:
                        # Successful futures return only their small coverage/provenance record.
                        result = future.result()
                    # A process failure is different from a legitimately nontrading symbol-day.
                    except Exception as error:
                        # Include the failure in the final denominator.
                        result = dict(job,status="error",error=repr(error))
                    # Retain all planned outcomes for the quality report.
                    results.append(result)
                    # Completed workers no longer belong in the rotating active heartbeat.
                    status.pop(job["symbol"]+":"+job["date"],None)
    # Preserve extraction evidence even if model fitting cannot proceed.
    save(args.output_dir/"build_summary.json",results)
    # Flatten depth/horizon coverage into an inspectable table for thin-name exclusions.
    coverage = [dict(symbol=item["symbol"],date=item["date"],**row) for item in results for row in item.get("coverage",[])]
    # Never require the user to infer missing-label coverage from model success alone.
    pd.DataFrame(coverage).to_csv(args.output_dir/"analysis"/"coverage.csv",index=False)
    # Stop model claims if any extraction failed unexpectedly.
    failed = [item for item in results if item["status"] == "error"]
    # Preserve a truthful final result even when analysis or plotting raises an exception.
    analysis,analysis_error = {},None
    # Missing-but-valid cells can be analysed with explicit conditional coverage.
    if not failed:
        # Isolate analysis failures from the completed immutable feature store.
        try:
            # Run the serial walk-forward study only after all feature jobs have finished.
            analysis = analyse(args.output_dir,results,settings,reporter)
        # Plotting or insufficient coverage must not silently produce a success verdict.
        except Exception:
            # Save the precise diagnostic without discarding completed feature partitions.
            analysis_error = traceback.format_exc()
    # Recheck every completed feature partition after analysis as part of provenance.
    changed_features = [item["feature_file"] for item in results if item["status"]=="ok" and (not (args.output_dir/item["feature_file"]).is_file() or digest(args.output_dir/item["feature_file"])!=item["feature_sha256"])]
    # Rehash inputs after the whole experiment, including model fitting.
    after = freeze([path for path in paths if path.is_file()],reporter,"Final hashing")
    # Compare exact raw/source bytes and missing files against their frozen versions.
    changed = [path for path,expected in hashes.items() if after.get(path)!=expected]+changed_features
    # New/deleted raw files also invalidate a frozen partition inventory.
    inventory = {path for date in dates for table in ("trades","ob_updates","ob_snapshot") for path in (C.PARSED_ROOT/table/f"date={date}").rglob("*.parquet")}
    # Retain all added/removed paths, not just those present before dispatch.
    changed += [str(path) for path in data_paths.symmetric_difference(inventory)]
    # Newly added or deleted local modules also change the tested source closure.
    after_sources = {path for path in code_root.rglob("*.py") if not {"docs","__pycache__",".git",".codex",".agents"}.intersection(path.relative_to(code_root).parts)}
    # Report source inventory changes instead of ignoring modules absent from initial hashes.
    changed += [str(path) for path in source_paths.symmetric_difference(after_sources)]
    # Completion certifies an intact research run, never a discovered profitable strategy.
    summary = dict(completed=not failed and analysis_error is None and not changed,planned=len(jobs),built=sum(item["status"]=="ok" for item in results),unavailable=sum(item["status"]=="unavailable" for item in results),failed=len(failed),changed_inputs=changed,analysis_error=analysis_error,analysis=analysis,live_approved=False,signal_approved=False)
    # Save final coverage and provenance only after every stage is assessed.
    save(args.output_dir/"summary.json",summary)
    # Completion is one immediate final message, not a burst of per-worker lines.
    reporter.close()
    # Keep completion concise; detailed failures and provenance remain in summary.json.
    print(f"[{datetime.now():%H:%M:%S}] Complete: {summary["completed"]} | Built {summary["built"]}/{len(jobs)} | Failed {len(failed)} | Held-out dates {analysis.get("heldout_dates",0)} | Report: {args.output_dir / 'summary.json'}",flush=True)
    # A failed/incomplete experiment must stop a chained terminal command.
    return 0 if summary["completed"] else 1


# Spawn workers only when explicitly invoked as the user-run research entry point.
if __name__ == "__main__":
    # Propagate the integrity/coverage result to the terminal.
    raise SystemExit(main())
