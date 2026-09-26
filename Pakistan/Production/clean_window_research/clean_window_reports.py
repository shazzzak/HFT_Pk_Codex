# Read saved cells without rerunning any strategies.
import json
# Keep report paths outside the repository.
from pathlib import Path
# Aggregate explicit arm and opening-cohort records.
import pandas as pd
# Save a strict final run summary.
from stock_search_util import save
# Retain the same additive money-component definitions.
from stock_search_accounting import MONEY

# Produce research-only P&L and coverage reports from durable per-cell evidence.
def report(output,jobs):
    # Normalize the caller's external output directory.
    output=Path(output)
    # Preserve all requested cells, including unavailable and failed observations.
    coverage,profits,buckets,windows,failures=[],[],[],[],[]
    # Count every requested stock-day exactly once.
    for job in jobs:
        # Read the completed or failed worker result.
        row=json.loads((output/(job['symbol']+'_'+job['date'])/'result.json').read_text())
        # Keep source failures explicit instead of adding zero-profit rows.
        if not row['passed']:
            # Preserve the exact failure for debugging.
            failures.append(dict(symbol=job['symbol'],date=job['date'],error=row['error']))
            # No valid cash summary exists for this cell.
            continue
        # Record the requested and accepted calendar duration side by side.
        coverage.append(dict(symbol=job['symbol'],date=job['date'],requested_minutes=row['requested_ms']/60000,source_window_minutes=row['source_window_ms']/60000,matched_minutes=row['accepted_ms']/60000,source_windows=len(row['windows']),matched_windows=sum(w['accepted'] for w in row['windows']),exit_excluded_windows=sum(not w['accepted'] for w in row['windows'])))
        # Preserve individual window boundaries and exclusion diagnostics.
        for window in row['windows']:
            # Nested exit reasons remain readable as JSON in the CSV cell.
            windows.append(dict(symbol=job['symbol'],date=job['date'],**window))
        # Lack of a matched clean interval is unavailable P&L, not a zero trading day.
        if row['arms'] is None:
            # Keep it in coverage but out of conditional profit totals.
            continue
        # Emit all twelve candidates for every accepted cell.
        for arm,value in row['arms'].items():
            # Preserve quantities alongside profits and coverage.
            profits.append(dict(symbol=job['symbol'],date=job['date'],arm=arm,net_pkr=value['net_pkr'],fills=value['fills'],matched_minutes=row['accepted_ms']/60000,max_inventory=value['max_inventory'],max_window_drawdown_pkr=value['max_window_drawdown_pkr']))
            # Keep every requested opening-time cohort in the detailed table.
            for bucket in value['buckets'].values():
                # Label the actual opening cohort independently of window end time.
                buckets.append(dict(symbol=job['symbol'],date=job['date'],arm=arm,**bucket))
    # Always emit coverage and failures even if no price window qualified.
    pd.DataFrame(coverage).to_csv(output/'coverage.csv',index=False)
    # Preserve unsuccessful exits with positions and pending messages.
    pd.DataFrame(windows).to_csv(output/'windows.csv',index=False)
    # Keep programming/source-contract failures in a separate actionable file.
    pd.DataFrame(failures).to_csv(output/'failures.csv',index=False)
    # Retain stock-day and opening-cohort detail without another long run.
    profit=pd.DataFrame(profits)
    # Never create a fabricated all-zero profit chart for missing coverage.
    totals={}
    # Produce money summaries only if at least one common window was accepted.
    if not profit.empty:
        # Preserve exact stock/date results under a deliberately qualified filename.
        profit.to_csv(output/'clean_window_profit_by_stock_day.csv',index=False)
        # Sum cash across independent windows, without annualizing missing periods.
        comparison=profit.groupby('arm',as_index=False).agg(net_pkr=('net_pkr','sum'),fills=('fills','sum'),matched_stock_days=('symbol','size'),matched_minutes=('matched_minutes','sum'))
        # Retain the totals in the machine-readable summary too.
        totals=dict(zip(comparison.arm,comparison.net_pkr))
        # Compare every candidate against the same base OBI arm on matched windows.
        comparison['difference_vs_OBI_best_pkr']=comparison.net_pkr-totals['OBI_best']
        # Save the twelve-way conditional comparison.
        comparison.to_csv(output/'comparison.csv',index=False)
        # Preserve per-stock performance without selecting its winner from these same observations.
        profit.groupby(['symbol','arm'],as_index=False).net_pkr.sum().to_csv(output/'profit_by_stock.csv',index=False)
        # Keep daily totals clearly labelled as usable-window profit, not full-day performance.
        daily=profit.groupby(['date','arm'],as_index=False).net_pkr.sum()
        # Retain all observed date totals and source coverage separately.
        daily.to_csv(output/'daily_profit.csv',index=False)
        # Group dates into months without extrapolating excluded periods.
        daily['month']=daily.date.str[:7]
        # Save actual summed clean-window cash by month.
        daily.groupby(['month','arm'],as_index=False).net_pkr.sum().to_csv(output/'monthly_profit.csv',index=False)
        # Save all monetary components by original acquisition-time cohort.
        bucket_frame=pd.DataFrame(buckets)
        # Preserve stock/date detail for loss audits.
        bucket_frame.to_csv(output/'pnl_by_opening_bucket_detail.csv',index=False)
        # Produce the compact requested capture/markout/closing-cost/fee decomposition.
        bucket_frame.groupby(['arm','bucket'],as_index=False)[list(MONEY)].sum().to_csv(output/'pnl_by_opening_bucket.csv',index=False)
        # Compare the original assigned portfolio under each fixed signal depth.
        assigned=[]
        # Avoid scanning the entire profit table for each of twenty thousand stock-days.
        indexed=profit.set_index(['symbol','date','arm']).net_pkr
        # Map known assignment names explicitly; do not guess unknown labels.
        styles={'OBI':'OBI','QT_2t@15':'QT15','QT_2t@20':'QT20'}
        # Use only stock-days with accepted windows for the conditional portfolio comparison.
        for job in jobs:
            # Find the exact incumbent style for this stock.
            label=job['assignment']
            # DROP contributes no incumbent trade; all twelve research candidates still ran above.
            if label=='DROP':
                # Do not silently claim DROP stocks were absent from the experiment.
                continue
            # Unknown assignment names must stop reporting rather than choose a default.
            if label not in styles:
                # Preserve the unsupported value for correction.
                raise ValueError('Unknown assigned portfolio label: '+str(label))
            # Select the already computed matched stock-day rows.
            # Keep each fixed signal setting distinct.
            for signal in ('best','w3','w4','w5'):
                # Fetch the corresponding incumbent-style candidate.
                key=(job['symbol'],job['date'],styles[label]+'_'+signal)
                # Do not convert missing source coverage into a zero profit.
                if key in indexed.index:
                    # Retain conditional portfolio contribution and identifying fields.
                    assigned.append(dict(symbol=job['symbol'],date=job['date'],signal=signal,net_pkr=float(indexed.loc[key])))
        # Save both portfolio contributions and fixed-depth totals.
        assigned=pd.DataFrame(assigned)
        # An all-DROP universe would have no incumbent trading contribution.
        if not assigned.empty:
            # Preserve reviewable stock/day contributions.
            assigned.to_csv(output/'assigned_portfolio_detail.csv',index=False)
            # Compare fixed-depth portfolios without choosing stock-specific winners retrospectively.
            assigned.groupby('signal',as_index=False).net_pkr.sum().to_csv(output/'assigned_portfolio_comparison.csv',index=False)
        # Use a noninteractive plotting backend for terminal jobs.
        import matplotlib
        # Do not open GUI windows on a long unattended run.
        matplotlib.use('Agg')
        # Render directly from the saved profit values.
        import matplotlib.pyplot as plt
        # Keep both profit and coverage visible so missing intervals cannot be overlooked.
        fig,axes=plt.subplots(1,2,figsize=(15,5))
        # Show conditional profit for the twelve unchanged candidate settings.
        axes[0].bar(comparison.arm,comparison.net_pkr)
        # Make long labels legible.
        axes[0].tick_params(axis='x',rotation=75)
        # State exactly what the profit represents.
        axes[0].set(title='Independent clean-window profit; not full-day P&L',ylabel='PKR after fees and window exits')
        # Compare accepted and excluded source duration over the same requested scope.
        accepted=sum(c['matched_minutes'] for c in coverage)
        # Count failed cells' requested time in the coverage denominator too.
        requested=sum(sum(b-a for a,b in j['params']['session_segments'])/60000 for j in jobs)
        # Show all excluded or unknown time rather than normalize it away.
        axes[1].bar(['Matched windows','Excluded / failed'],[accepted,requested-accepted])
        # Identify the summed stock-time unit.
        axes[1].set(title='Coverage across requested stocks and dates',ylabel='Stock-minutes')
        # Avoid clipping descriptive labels.
        fig.tight_layout()
        # Write one portable review image alongside the CSV evidence.
        fig.savefig(output/'profit_and_coverage.png',dpi=160)
        # Release plotting memory before the process exits.
        plt.close(fig)
    # Count fully closed usable windows and all failed exit windows.
    exit_exclusions=sum(c['exit_excluded_windows'] for c in coverage)
    # Publish only claims supported by these conditional source windows.
    summary=dict(completed=not failures,experiment='retrospective clean-window research',full_day_pnl=False,live_approved=False,requested_stock_days=len(jobs),failed_stock_days=len(failures),stock_days_with_matched_windows=sum(c['matched_minutes']>0 for c in coverage),matched_windows=sum(c['matched_windows'] for c in coverage),exit_excluded_windows=exit_exclusions,matched_minutes=sum(c['matched_minutes'] for c in coverage),requested_minutes=sum(sum(b-a for a,b in j['params']['session_segments'])/60000 for j in jobs),net_pkr=totals,selection_warning='Window endpoints are known retrospectively. Flat restarts and early exit buffers are research assumptions. Missing periods are not zero profit. Failed closes are excluded across all arms; nonzero exit exclusions prevent an unqualified ranking.')
    # Save the final summary atomically after all evidence files are written.
    save(output/'summary.json',summary)
    # Return the same facts for the terminal's final line.
    return summary
