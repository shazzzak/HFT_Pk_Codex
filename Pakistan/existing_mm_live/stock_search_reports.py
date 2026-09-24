# Stream large per-cell evidence without retaining all fill records in memory.
import csv
# Read completed cell checkpoints.
import json
# Aggregate opening-cohort components incrementally.
from collections import defaultdict
# Resolve output and input paths.
from pathlib import Path
# Use tabular summaries only after the expensive replay has finished.
import pandas as pd
# Share the exact money-column contract with FIFO accounting.
from stock_search_accounting import MONEY
# Publish strict summary evidence.
from stock_search_util import save

# Produce matched comparisons, opening-time attribution and dated selections.
def report(output, jobs, assignment, qualification):
    # Normalize the results directory.
    output = Path(output)
    # Accumulate smaller month/stock/bucket totals while streaming day-level rows.
    aggregate = defaultdict(lambda: defaultdict(float))
    # Retain only compact per-stock/day/configuration metrics for selection.
    metrics = []
    # Write detailed opening cohorts directly to disk.
    with (output / 'pnl_by_opening_bucket.csv').open('w', newline='') as stream:
        # Initialize the writer after the first complete row determines columns.
        writer = None
        # Visit every predeclared stock-day, including previously unprofitable names.
        for job in jobs:
            # Load the successfully reconciled cell summary.
            row = json.loads((output / f"{job['symbol']}_{job['date']}_seed0.json").read_text())
            # Never silently exclude a failed stock-day.
            if not row['passed']:
                # Require repair and identical-input resume first.
                raise ValueError('Cannot report incomplete search')
            # Preserve each of the twelve logical configurations, including aliases.
            for arm, result in row['arms'].items():
                # Retain metrics needed for risk and profit comparisons.
                metrics.append(dict(symbol=job['symbol'], date=job['date'], month=job['date'][:7], arm=arm, net_pkr=result['net_pkr'], fills=result['fills'], shares=result['shares'], residual_shares=result['eod']['unfilled_sh'], max_drawdown_pkr=result['max_drawdown_pkr'], max_abs_inventory=result['max_abs_inventory'], capacity_reductions=result['capacity_reductions'], unattributed_qty=sum(r['unattributed_qty'] for r in result['attribution'])))
                # Persist acquisition-cohort accounting at the finest requested level.
                for component in result['attribution']:
                    # Attach stock, date and configuration identity.
                    record = dict(symbol=job['symbol'], date=job['date'], arm=arm, **component)
                    # Initialize an explicitly labelled stable schema.
                    if writer is None:
                        # Use the complete first row's columns.
                        writer = csv.DictWriter(stream, fieldnames=list(record))
                        # Write one header for the whole file.
                        writer.writeheader()
                    # Stream the full detailed row.
                    writer.writerow(record)
                    # Group by original inventory-acquisition bucket, never exit time.
                    key = (job['symbol'], job['date'][:7], arm, component['bucket'])
                    # Accumulate all numeric component and quantity fields.
                    for name, value in component.items():
                        # Leave the bucket label in the grouping key.
                        if name != 'bucket':
                            # Preserve signed P&L and positive fees separately.
                            aggregate[key][name] += value
    # Materialize compact metrics only, not all fill records or detailed cohorts.
    table = pd.DataFrame(metrics)
    # Persist every stock-day/configuration including negative outcomes.
    table.to_csv(output / 'comparison.csv', index=False)
    # Form the smaller opening-cohort monthly table.
    buckets = pd.DataFrame([dict(symbol=s, month=m, arm=a, bucket=b, **v) for (s, m, a, b), v in aggregate.items()])
    # Retain each stock's month-by-month attribution.
    buckets.to_csv(output / 'pnl_by_stock_month_bucket.csv', index=False)
    # Keep a simple twelve-by-four total component table for review.
    buckets.groupby(['arm', 'bucket']).sum(numeric_only=True).to_csv(output / 'pnl_by_bucket.csv')
    # Save matched daily totals for every fixed configuration.
    daily = table.groupby(['date', 'arm']).net_pkr.sum().unstack()
    # Retain all daily losses as well as gains.
    daily.to_csv(output / 'daily_profit.csv')
    # Save monthly totals for the same unchanged configurations.
    monthly = table.groupby(['month', 'arm']).net_pkr.sum().unstack()
    # Expose changes across market conditions.
    monthly.to_csv(output / 'monthly_profit.csv')
    # Show the complete profit comparison stock by stock.
    table.groupby(['symbol', 'arm']).net_pkr.sum().unstack().to_csv(output / 'profit_by_stock.csv')
    # Translate assigned labels without reviving DROP in the incumbent portfolio.
    styles = {'OBI': 'OBI', 'QT_2t@15': 'QT15', 'QT_2t@20': 'QT20'}
    # Collect four portfolio comparisons on the same recorded assignment.
    portfolio = []
    # Preserve the best-level assigned baseline plus its three weighted variants.
    for signal in ('best', 'w3', 'w4', 'w5'):
        # Select the original style for each stock; DROP remains zero in this comparator.
        selected = table[[assignment.get(s) in styles and a == styles[assignment[s]] + '_' + signal for s, a in zip(table.symbol, table.arm)]]
        # Keep every requested date, including any date with no selected trades.
        values = selected.groupby('date').net_pkr.sum().reindex(sorted(table.date.unique()), fill_value=0)
        # Label the comparator without calling retrospectively selected settings causal.
        portfolio.extend(dict(date=d, portfolio='assigned_' + signal, net_pkr=float(v)) for d, v in values.items())
    # Export the requested assigned-portfolio profit comparison.
    pd.DataFrame(portfolio).to_csv(output / 'assigned_portfolio_daily.csv', index=False)
    # Select only from earlier months using the independently tested rule.
    selections, following = select_prior_months(table)
    # Record choices even when this pilot has too little history to select anything.
    pd.DataFrame(selections, columns=['symbol', 'month', 'training_months', 'chosen', 'training_net_pkr']).to_csv(output / 'monthly_selections.csv', index=False)
    # Preserve selected results separately from fixed-configuration comparisons.
    pd.DataFrame(following, columns=['symbol', 'date', 'arm', 'net_pkr']).to_csv(output / 'selected_following_month_profit.csv', index=False)
    # Generate readable profit images without another market-data scan.
    import matplotlib
    # Use terminal-safe rendering.
    matplotlib.use('Agg')
    # Import plotting only after choosing the backend.
    import matplotlib.pyplot as plt
    # Show accumulated daily net profit for all fixed configurations.
    daily.cumsum().plot(figsize=(14, 7), ylabel='Cumulative net profit (PKR)')
    # Keep labels and legend inside the saved figure.
    plt.tight_layout()
    # Save the figure next to its underlying daily table.
    plt.savefig(output / 'cumulative_profit.png', dpi=140)
    # Release plotting state.
    plt.close()
    # Show the four opening cohorts with signed components separately from costs.
    components = buckets.groupby(['arm', 'bucket'])[list(MONEY[:-1])].sum()
    # Plot each candidate in its own panel to avoid overlapping labels.
    figure, axes = plt.subplots(4, 3, figsize=(18, 18))
    # Align one panel with each of the twelve predeclared configurations.
    for axis, arm in zip(axes.flat, daily.columns):
        # Use component columns from the exported accounting table.
        components.loc[arm].plot.bar(ax=axis, title=arm, legend=False, fontsize=8)
    # Provide one shared component legend.
    figure.legend(*axes.flat[0].get_legend_handles_labels(), loc='upper center', ncol=3)
    # Leave room for the shared legend.
    figure.tight_layout(rect=(0, 0, 1, .95))
    # Save the component breakdown image.
    figure.savefig(output / 'opening_bucket_components.png', dpi=120)
    # Free all figure memory.
    plt.close(figure)
    # Return explicit totals, denominators and attribution completeness.
    summary = dict(completed=True, stock_days=len(jobs), configurations=12, calibration_status=qualification, independent_validation=False, live_approved=False, attribution_complete=bool(table.unattributed_qty.sum() == 0), unattributed_qty=float(table.unattributed_qty.sum()), profit={a: dict(total_pkr=float(daily[a].sum()), average_daily_pkr=float(daily[a].mean()), average_monthly_pkr=float(monthly[a].mean())) for a in daily}, selection_rule='Prior three months; >=40 dates, >=2 positive months, positive net, zero closing residual shares and zero unattributed matched shares. Research eligibility only; not approved live risk limits.')
    # Publish a strict final report.
    save(output / 'summary.json', summary)
    # Let the entry point print a concise success statement.
    return summary

# Separate historical selection from report rendering for chronology tests.
def select_prior_months(table):
    # Retain candidate selections and genuinely following-month outcomes separately.
    selections, following = [], []
    # Use each full month only after three earlier evaluation months exist.
    for month in sorted(table.month.unique()):
        # Construct prior three calendar months without looking at their ranking today.
        period = pd.Period(month, freq='M')
        # Require complete coverage of each of the three training months.
        previous = [str(period - i) for i in (3, 2, 1)]
        # October-December establish initial training, not reported selected profits.
        if not set(previous).issubset(set(table.month)):
            # Never train a selection using its own evaluation month.
            continue
        # Evaluate every stock, irrespective of its prior assigned label.
        for symbol in sorted(table.symbol.unique()):
            # Access only the earlier three months for choosing the setting.
            train = table[(table.symbol == symbol) & table.month.isin(previous)]
            # Aggregate the predeclared eligibility rules by configuration.
            scores = train.groupby('arm').agg(net=('net_pkr', 'sum'), residual=('residual_shares', 'sum'), unattributed=('unattributed_qty', 'sum'), days=('date', 'nunique'))
            # Count positive training months; this is fixed before examining results.
            positive = train.groupby(['arm', 'month']).net_pkr.sum().gt(0).groupby('arm').sum()
            # Use conservative research eligibility, not invented live risk limits.
            allowed = scores[(scores.net > 0) & (scores.residual == 0) & (scores.unattributed == 0) & (scores.days >= 40) & (positive >= 2)]
            # Prefer fewer levels and no queue lean on an exact profit tie.
            rank = {a: i for i, a in enumerate(('OBI_best', 'QT15_best', 'QT20_best', 'OBI_w3', 'QT15_w3', 'QT20_w3', 'OBI_w4', 'QT15_w4', 'QT20_w4', 'OBI_w5', 'QT15_w5', 'QT20_w5'))}
            # Keep no-trade available when no setting satisfies the fixed checks.
            choice = min(allowed.index, key=lambda a: (-allowed.loc[a, 'net'], rank[a])) if len(allowed) else 'NO_TRADE'
            # Save the information cutoff and exact training score used.
            selections.append(dict(symbol=symbol, month=month, training_months=' '.join(previous), chosen=choice, training_net_pkr=float(allowed.loc[choice, 'net']) if choice != 'NO_TRADE' else 0.0))
            # Read the following month's outcomes only after the choice is frozen.
            test = table[(table.symbol == symbol) & (table.month == month)]
            # Preserve no-trade days as zero rather than dropping them.
            for date in sorted(test.date.unique()):
                # Match exactly one candidate on this evaluation day.
                profit = float(test[(test.date == date) & (test.arm == choice)].net_pkr.iloc[0]) if choice != 'NO_TRADE' else 0.0
                # Store the following-month chosen-portfolio result.
                following.append(dict(symbol=symbol, date=date, arm=choice, net_pkr=profit))
    # Return choices and following-month results without altering the inputs.
    return selections, following
