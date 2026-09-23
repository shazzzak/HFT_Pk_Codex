"""Date-walk-forward density research with equal-name/day summaries; no trading decisions."""
# Keep artifacts under the explicitly chosen external run directory.
from pathlib import Path
# Store training-only coefficients and transforms for reproducibility.
import json
# Fit fixed linear/ridge models and bootstrap whole dates.
import numpy as np
# Assemble held-out date metrics and explicit geometry/liquidity strata.
import pandas as pd
# Use the same fixed depth windows and geometry thresholds as extraction.
from book_density import DEPTHS, geometry_group


# Fit scaling, clipping and coefficients exclusively on earlier trading dates.
def predict(train, test, columns):
    # Extract finite matrices after the caller chooses identical model rows.
    x, z = train[columns].to_numpy(float), test[columns].to_numpy(float)
    # Each training date has equal weight despite differences in valid row counts.
    weights = 1/train.groupby("date")["date"].transform("size").to_numpy(float)
    # Normalize the weighted objective to one so ridge strength is reproducible.
    weights /= weights.sum()
    # Learn feature clipping from training observations only.
    low, high = np.quantile(x, [.005,.995], axis=0)
    # Apply exactly the same learned transformation to unseen test observations.
    x, z = np.clip(x,low,high), np.clip(z,low,high)
    # Estimate training means and scales without using held-out dates.
    mean = np.sum(x*weights[:,None],axis=0)
    # Constant controls stay zero after scaling instead of destabilizing the fit.
    scale = np.sqrt(np.sum((x-mean)**2*weights[:,None],axis=0))
    # Replace only numerically constant scales, retaining every declared coefficient.
    scale[scale < 1e-10] = 1
    # Include an unpenalized intercept in both design matrices.
    x, z = np.c_[np.ones(len(x)),(x-mean)/scale], np.c_[np.ones(len(z)),(z-mean)/scale]
    # Preserve raw markout tails; do not winsorize the outcome using future data.
    y = train.target.to_numpy(float)
    # Fix regularization before seeing results; there is no in-sample tuning search.
    penalty = np.eye(x.shape[1])*1e-4
    # The model intercept should reflect the earlier-date target mean.
    penalty[0,0] = 0
    # Solve the weighted ridge objective with only declared past inputs.
    beta = np.linalg.solve(x.T@(weights[:,None]*x)+penalty, x.T@(weights*y))
    # Retain training outcome scale for equal-name normalized loss comparisons.
    ymean = float(weights@y)
    # A zero-variance training target cannot establish predictable markout variation.
    variance = float(weights@((y-ymean)**2))
    # Return predictions and the entire transform needed to reproduce them.
    return z@beta, dict(columns=columns,clip_low=low.tolist(),clip_high=high.tolist(),mean=mean.tolist(),scale=scale.tolist(),beta=beta.tolist(),train_mean=ymean,train_variance=variance)


# Assign liquidity from strictly prior turnover within the explicitly selected universe.
def liquidity_labels(metadata, window, min_train):
    # Missing symbol-days remain absent rather than silently zero-volume observations.
    table = pd.DataFrame([row for row in metadata if row.get("status") != "error" and row.get("turnover") is not None])
    # A usable run needs at least one observed instrument/day.
    if table.empty:
        # The caller reports missing data rather than pretending to classify names.
        return {}
    # Date ordering makes the shift's no-lookahead direction explicit.
    table = table.sort_values(["symbol","date"])
    # Each name's scale uses only prior observed trading days.
    table["prior_turnover"] = table.groupby("symbol").turnover.transform(lambda values: values.shift(1).rolling(window,min_periods=min_train).median())
    # Freeze one label per date/name without inspecting any forward return.
    labels = {}
    # Tercile boundaries are contemporaneously available prior-data statistics.
    for date, part in table.groupby("date"):
        # Names lacking sufficient earlier observations are explicitly unclassified.
        available = part.dropna(subset=["prior_turnover"])
        # A tiny cross-section cannot support meaningful terciles.
        if len(available) < 3:
            # Preserve insufficient coverage rather than inventing buckets.
            continue
        # Use dollar/PKR turnover rather than absolute share count across price levels.
        lower, upper = available.prior_turnover.quantile([1/3,2/3])
        # Apply deterministic thresholds, with ties retained together.
        for row in available.itertuples():
            # Low-turnover means relative to this declared research universe only.
            labels[(row.symbol,date)] = "low_turnover" if row.prior_turnover <= lower else "high_turnover" if row.prior_turnover >= upper else "middle_turnover"
    # Return classification independent of strategy assignment labels.
    return labels


# Evaluate one symbol with a purged-by-date rolling training window.
def evaluate_symbol(frame, horizons, min_train, window, liquidity, progress=None):
    # Every feature partition is tagged by its actual calendar session.
    dates = sorted(frame.date.unique())
    # Retain daily model errors, training-frozen deciles and fit provenance.
    daily, deciles, fits = [], [], []
    # Do not train or validate across the same trading date.
    for index in range(min_train,len(dates)):
        # Limit training to the declared number of earlier sessions.
        train_dates = dates[max(0,index-window):index]
        # The whole next session is held out; labels never cross session boundaries.
        past, today = frame[frame.date.isin(train_dates)], frame[frame.date == dates[index]]
        # Identify the instrument once; never mix names inside a fitted model.
        symbol = str(today.symbol.iloc[0])
        # Publish progress without printing from inner fitting loops.
        if progress:
            # This count represents completed held-out dates, not individual regressions.
            progress(symbol,dates[index],index-min_train,len(dates)-min_train)
        # Compare every requested level window with its own same-depth OBI baseline.
        for depth in DEPTHS:
            # Fixed-depth and all-visible measures retain their exact column names.
            key = str(depth) if depth is not None else "all"
            # Establish causal, symmetric scale controls alongside L1 and same-depth OBI.
            base = ["obi_1",f"obi_{key}","obi_squared","obi_cubed","spread_bps",f"log_depth_{key}",f"log_span_{key}",f"n_bid_{key}",f"n_ask_{key}","realized_vol_bps","ofi_ewma","quote_age_ms"]
            # Compare the requested composite with a genuinely volume-free geometry term.
            variants = {"density": [f"density_{key}"], "inclusive_density": [f"density_inclusive_{key}"], "geometry": [f"geometry_{key}","obi_geometry"]}
            # New feature stores can test price-distance OBI beyond the existing baseline.
            if f"pd_obi_{key}" in frame.columns:
                # Keep the original primary geometry family unchanged; this arm is exploratory.
                variants["price_distance"] = [f"pd_obi_{key}"]
            # New concentration stores support a pure concentration arm and a location extension.
            if f"hhi_normalized_bid_{key}" in frame.columns:
                # Do not interpret a bid-minus-ask HHI difference as inherently directional.
                concentration = [f"hhi_normalized_{side}_{key}" for side in ("bid","ask")]
                # Test concentration alone against the same existing OBI/control baseline.
                variants["concentration"] = concentration
                # Retain side-specific location rather than assigning a trust label to a wall.
                variants["concentration_location"] = concentration+[f"{metric}_{side}_{key}" for metric in ("touch_share","log_hhi_distance") for side in ("bid","ask")]
            # New liquidity stores retain coverage-aware cost and observed flow comparisons.
            if "walk_buy_1x_coverage" in frame.columns:
                # Observed partial cost is never represented as a fully executable order cost.
                variants["observed_walk"] = [f"walk_{side}_{multiple}x_{metric}" for side in ("buy","sell") for multiple in (1,3,5) for metric in ("partial_impact_bps","coverage")]
                # Partial observation windows retain an explicit duration control.
                variants["attributed_flow"] = ["dyn_observed_ms"]+[f"dyn_{side}_{metric}_shares" for side in ("buy","sell") for metric in ("added","added_at_consumed_price","executed","cancelled","unresolved_trade")]
                # Retain observed depth separately from whether its price extent covers the band.
                variants["band_depth"] = [f"depth_observed_pkr_{side}_{band}t" for side in ("bid","ask") for band in (1,5,20)]+[f"depth_extent_covers_{side}_{band}t" for side in ("bid","ask") for band in (1,5,20)]
            # Every model must use exactly the same valid rows for an honest comparison.
            required = list(dict.fromkeys(base+[item for cols in variants.values() for item in cols]))
            # Fixed clock horizons answer the immediate-markout question directly.
            for horizon in horizons:
                # Work on independent frames so feature transforms cannot leak across fits.
                tr, te = past.copy(), today.copy()
                # Materialize nonlinear OBI controls and interaction terms causally.
                for part in (tr,te):
                    # A flexible baseline avoids mistaking a simple OBI transform for new data.
                    part["obi_squared"] = part[f"obi_{key}"]**2
                    # Add a fixed cubic without selecting polynomial degree on test results.
                    part["obi_cubed"] = part[f"obi_{key}"]**3
                    # Geometry's interaction allows a change in the slope of quantity pressure.
                    part["obi_geometry"] = part[f"obi_{key}"]*part[f"geometry_{key}"]
                    # Copy the already-guarded target without rebuilding labels from sampled rows.
                    part["target"] = part[f"markout_{horizon}ms_bps"]
                # Undefined depth or labels must be counted as excluded, never set to neutral.
                tr,te = (part.dropna(subset=required+["target"]) for part in (tr,te))
                # Require meaningful training data and several observations in the held-out day.
                if len(tr) < 200 or len(te) < 20 or tr.date.nunique() < min_train:
                    # Coverage files expose sparse/unavailable depths separately.
                    continue
                # Fit the declared OBI/control baseline once for all same-row comparisons.
                baseline, baseline_fit = predict(tr,te,base)
                # Flat training labels cannot support normalized predictive comparisons.
                if baseline_fit["train_variance"] <= 1e-12:
                    # Do not manufacture enormous R-squared gains from zero variance.
                    continue
                # Fixed geometry thresholds avoid fitting strata to favorable outcomes.
                geometry = geometry_group(te[f"geometry_{key}"].to_numpy())
                # Neutral touch imbalance is an explicit subset, not the entire hypothesis test.
                neutral = np.abs(te.obi_1.to_numpy()) <= .1
                # Preserve the same raw held-out targets for each model.
                y = te.target.to_numpy()
                # Each augmented model adds only its preregistered geometry/composite columns.
                for variant, additions in variants.items():
                    # All clipping, scaling and coefficient estimation use past dates only.
                    prediction, fitted = predict(tr,te,base+additions)
                    # Freeze enough context to reconstruct this particular fit later.
                    identity = dict(symbol=symbol,date=dates[index],depth=key,horizon_ms=horizon,variant=variant)
                    # Preserve the complete baseline and augmented transforms and training window.
                    fits.append(dict(identity,train_dates=train_dates,train_rows=len(tr),test_rows=len(te),baseline=baseline_fit,augmented=fitted))
                    # Error reduction measures increment beyond same-depth OBI on held-out rows.
                    gain = (y-baseline)**2-(y-prediction)**2
                    # Report pooled rows and each geometry group, with explicit overlapping subsets.
                    for group in ("ALL","equal_span","moderate_asymmetry","ask_span_gt_2x","bid_span_gt_2x"):
                        # Do not force the dense majority to dominate sparse-book inference.
                        geometry_mask = np.ones(len(te),dtype=bool) if group == "ALL" else geometry == group
                        # The neutral-L1 subset is additional evidence, not a replacement for controls.
                        for subset in ("all_l1","neutral_l1"):
                            # Select using observed geometry and imbalance only.
                            mask = geometry_mask & (neutral if subset == "neutral_l1" else True)
                            # Tiny cells receive coverage counts but no unstable effect estimate.
                            if mask.sum() < 20:
                                # At least twenty rows are required per daily stratum.
                                continue
                            # Summaries later weight each name/date equally, not by event count.
                            daily.append(dict(identity,geometry=group,subset=subset,liquidity=liquidity.get((symbol,dates[index]),"unclassified"),rows=int(mask.sum()),gain_bps2=float(gain[mask].mean()),normalized_gain=float(gain[mask].mean()/baseline_fit["train_variance"]),baseline_mse=float(((y-baseline)[mask]**2).mean()),augmented_mse=float(((y-prediction)[mask]**2).mean())))
                    # Use training quantiles for the feature itself; never rank on future outcomes.
                    feature = additions[0]
                    # Collapse ties explicitly; equal-span/neutral books can have few distinct bins.
                    edges = np.unique(np.quantile(tr[feature],np.linspace(0,1,11)[1:-1]))
                    # Place all out-of-training-range values into the extreme bins.
                    bins = np.searchsorted(edges,te[feature].to_numpy(),side="right")
                    # Preserve out-of-sample markout curves with zeros included.
                    for number in np.unique(bins):
                        # Each date/bin is an explicit unit for later equal-name/day aggregation.
                        mask = bins == number
                        # Save actual bin count so collapsed ties cannot masquerade as ten deciles.
                        deciles.append(dict(identity,bin=int(number),bins=len(edges)+1,rows=int(mask.sum()),feature_mean=float(te.loc[mask,feature].mean()),markout_mean_bps=float(y[mask].mean())))
    # Return model evidence even when no valid fits existed; the caller flags coverage.
    return pd.DataFrame(daily),pd.DataFrame(deciles),fits


# Approximate uncertainty by resampling consecutive blocks of entire trading dates.
def block_interval(values, repetitions=1000):
    # Remove missing stratum dates; inference is conditional on reported eligible dates.
    values = np.asarray(values,float)
    # Fewer than twenty held-out dates is descriptive evidence only.
    if np.isfinite(values).sum() < 20:
        # Avoid presenting narrow row-level intervals from dependent observations.
        return np.nan,np.nan,np.nan
    # Fix a five-date block and a seed before inspecting any outcomes.
    rng, length = np.random.default_rng(1729), min(5,len(values))
    # Sample overlapping non-circular blocks, preserving within-block daily dependence.
    starts = rng.integers(0,len(values)-length+1,size=(repetitions,int(np.ceil(len(values)/length))))
    # Join enough blocks for one equal-length resampled history.
    indices = (starts[:,:,None]+np.arange(length)).reshape(repetitions,-1)[:,:len(values)]
    # Every input value is already an equal-name mean for that trading date.
    sampled = values[indices]
    # Preserve missing calendar dates rather than compressing unrelated days together.
    counts = np.isfinite(sampled).sum(axis=1)
    # An all-missing resample cannot supply a bootstrap mean.
    means = np.nansum(sampled,axis=1)[counts>0]/counts[counts>0]
    # Sparse histories need enough nonempty replicates to support an interval.
    if len(means) < .95*repetitions:
        # Retain descriptive estimates while withholding unsupported inference.
        return np.nan,np.nan,np.nan
    # Center the empirical bootstrap distribution for an approximate two-sided null test.
    p = (1+np.count_nonzero(np.abs(means-means.mean()) >= abs(np.nanmean(values))))/(len(means)+1)
    # Basic bootstrap intervals retain the original sample mean as their center.
    low,high = np.quantile(means-means.mean(),[.025,.975])
    # Approximate block uncertainty is not a proof of causality or an execution model.
    return float(np.nanmean(values)-high),float(np.nanmean(values)-low),float(p)


# Summarize pooled, per-name and prior-liquidity results without liquid-name row dominance.
def summarize(daily):
    # No valid held-out model is an incomplete experiment, not zero predictive power.
    if daily.empty:
        # The runner surfaces this as insufficient coverage.
        return pd.DataFrame()
    # Declare every reporting dimension explicitly for multiple-testing adjustment.
    keys = ["depth","horizon_ms","variant","geometry","subset"]
    # Assemble overlapping descriptive scopes without treating them as independent tests.
    scopes = [("portfolio",daily)]
    # Per-name outputs make illiquid or anomalous instruments visible.
    scopes += [("symbol:"+str(symbol),part) for symbol,part in daily.groupby("symbol")]
    # These labels were frozen from strictly prior turnover, not selected by test P&L.
    scopes += [("liquidity:"+str(label),part) for label,part in daily.groupby("liquidity")]
    # Retain one record per declared hypothesis/reporting scope.
    rows = []
    # Use one shared trading-date calendar so missing strata do not shorten serial gaps.
    calendar = sorted(daily.date.unique())
    # Aggregate names before dates to preserve cross-sectional dependence.
    for scope,part in scopes:
        # Separate feature, horizon and geometry hypotheses.
        for key,group in part.groupby(keys,sort=True):
            # Each name/date contributes one mean regardless of its number of rows.
            dates = group.groupby("date",sort=True).normalized_gain.mean()
            # Resample whole dates rather than independent correlated feature rows.
            low,high,p = block_interval(dates.reindex(calendar).to_numpy())
            # Preserve the unstandardized loss gain for economic scale interpretation.
            raw = group.groupby("date").gain_bps2.mean().mean()
            # Include eligible rows/names/dates so conditional coverage is never hidden.
            rows.append(dict(zip(keys,key),scope=scope,mean_normalized_gain=float(dates.mean()),mean_gain_bps2=float(raw),ci_low=low,ci_high=high,p_approx=p,dates=len(dates),names=group.symbol.nunique(),rows=int(group.rows.sum()),inference="descriptive_only" if len(dates)<20 else "approximate_5_date_block_bootstrap"))
    # Separate a preregistered primary family from the broader exploratory screen.
    result = pd.DataFrame(rows)
    # Primary: pure geometry at 250ms, portfolio/low-turnover, pooled or strongly asymmetric.
    primary = (result.variant=="geometry") & (result.horizon_ms==250) & result.scope.isin(["portfolio","liquidity:low_turnover"]) & (result.subset=="all_l1") & result.geometry.isin(["ALL","ask_span_gt_2x","bid_span_gt_2x"])
    # This family is fixed before inspecting market data or estimated coefficients.
    result["family"] = np.where(primary,"primary_250ms_geometry","exploratory")
    # Descriptive-only cells cannot be assigned spurious significance probabilities.
    result["holm_p_approx"] = np.nan
    # Control each explicitly labeled family without presenting exploratory scans as confirmation.
    for family,group in result.groupby("family"):
        # Rank only hypotheses with enough held-out dates for the approximate block procedure.
        eligible = group.p_approx.dropna().sort_values()
        # Holm's running maximum preserves monotonic adjusted significance.
        adjusted = np.maximum.accumulate(eligible.to_numpy()*(len(eligible)-np.arange(len(eligible))))
        # Cap adjusted probabilities at one and retain missing inference elsewhere.
        result.loc[eligible.index,"holm_p_approx"] = np.minimum(1,adjusted)
    # No automatic signal promotion or strategy alteration follows a statistical result.
    return result


# Export standalone figures using a noninteractive backend and external cache.
def plot_results(summary,deciles,output):
    # Keep plotting optional until the user's environment has matplotlib available.
    import matplotlib
    # Headless execution must not open GUI windows during a long research run.
    matplotlib.use("Agg")
    # Load plotting only after the backend has been selected.
    import matplotlib.pyplot as plt
    # Compare held-out incremental geometry skill across horizons and depths.
    selected = summary[(summary.scope=="portfolio") & (summary.variant=="geometry") & (summary.geometry=="ALL") & (summary.subset=="all_l1")]
    # Always produce an informative plot, including clearly labeled missing results.
    fig,ax = plt.subplots(figsize=(8,5))
    # Keep every requested depth distinct, including all-known-price levels.
    for depth,part in selected.groupby("depth"):
        # Fixed ordering avoids connecting horizons in arbitrary file order.
        part = part.sort_values("horizon_ms")
        # Plot equal-name/date held-out gain, not in-sample correlation.
        ax.plot(part.horizon_ms,part.mean_normalized_gain,marker="o",label=f"depth {depth}")
    # Zero is the baseline's performance; negative increments are honest failures.
    ax.axhline(0,color="black",linewidth=.8)
    # Label the exact metric rather than implying tradable profit.
    ax.set(xlabel="Selected-clock horizon (ms)",ylabel="OOS loss reduction / training variance",title="Increment from spacing beyond OBI controls")
    # Show the sparse horizon spacing clearly.
    ax.set_xscale("log")
    # Avoid an empty misleading legend when coverage produced no eligible fits.
    if not selected.empty:
        # Match legend labels to frozen depth settings.
        ax.legend()
    # Save a shareable scientific figure outside the source repository.
    fig.tight_layout(); fig.savefig(Path(output)/"geometry_oos_gain.png",dpi=160); plt.close(fig)
    # Plot per-name geometry effects at the preregistered 250ms horizon and depth 3.
    selected = summary[summary.scope.str.startswith("symbol:") & (summary.variant=="geometry") & (summary.depth=="3") & (summary.horizon_ms==250) & (summary.subset=="all_l1")]
    # Separate book geometry groups so liquid balanced states do not hide sparse results.
    matrix = selected.pivot(index="scope",columns="geometry",values="mean_normalized_gain")
    # A missing stratum is different from zero estimated gain.
    if not matrix.empty:
        # Scale height to the number of tested names without clipping labels.
        fig,ax = plt.subplots(figsize=(9,max(4,len(matrix)*.22)))
        # Symmetric colors distinguish improvements from deterioration.
        bound = max(.001,float(np.nanmax(np.abs(matrix.to_numpy()))))
        # Missing values remain blank in the heat map.
        image = ax.imshow(matrix.to_numpy(),aspect="auto",cmap="RdBu",vmin=-bound,vmax=bound)
        # Keep actual instruments and geometry strata readable.
        ax.set_yticks(range(len(matrix)),matrix.index.str.replace("symbol:","",regex=False))
        # Rotate long group labels to prevent collisions.
        ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=25,ha="right")
        # Explicitly identify the exploratory effect, horizon and level window.
        ax.set_title("Geometry increment: depth 3, 250ms (held-out dates)")
        # Explain color magnitude in the figure itself.
        fig.colorbar(image,ax=ax,label="Normalized held-out loss reduction")
        # Export a standalone PNG with complete labels.
        fig.tight_layout(); fig.savefig(Path(output)/"geometry_by_symbol.png",dpi=160); plt.close(fig)
    # Raw markout curves complement model-dependent error reductions.
    selected = deciles[(deciles.depth=="3") & (deciles.horizon_ms==250)] if not deciles.empty else deciles
    # Do not manufacture curves where no held-out bins are available.
    if not selected.empty:
        # Use per-date/name bin means instead of pooling liquid-name rows.
        fig,ax = plt.subplots(figsize=(8,5))
        # Plot each standalone feature's observed held-out markout relation.
        for variant,part in selected.groupby("variant"):
            # Equal-weight names within dates, then dates, for each actual bin index.
            curve = part.groupby(["date","bin"])[["feature_mean","markout_mean_bps"]].mean().groupby("bin").mean()
            # Use feature value on the axis so collapsed quantile ties are visible.
            ax.plot(curve.feature_mean,curve.markout_mean_bps,marker="o",label=variant)
        # A flat zero line represents no average directional price change.
        ax.axhline(0,color="black",linewidth=.8)
        # Describe these as conditional means, never causal effects.
        ax.set(xlabel="Training-binned feature value",ylabel="Held-out 250ms mid markout (bps)",title="Depth-3 feature bins; zero returns included")
        # Name the distinct geometry/density feature definitions.
        ax.legend()
        # Persist the complementary raw-response visualization.
        fig.tight_layout(); fig.savefig(Path(output)/"heldout_markout_curves.png",dpi=160); plt.close(fig)

    # Show the new arm's held-out gains separately from the original geometry plots.
    selected = summary[(summary.variant=="price_distance") & (summary.scope=="portfolio") & (summary.geometry=="ALL") & (summary.subset=="all_l1")]
    # Older feature stores have no price-distance arm and should not produce an empty figure.
    if not selected.empty:
        # Compare horizons on one readable exportable chart.
        fig,ax = plt.subplots(figsize=(8,5))
        # Keep all requested depth definitions visibly distinct.
        for depth,part in selected.groupby("depth"):
            # Preserve ascending horizon order on each line.
            part=part.sort_values("horizon_ms")
            # Plot predictive loss improvement, not trading profit.
            ax.plot(part.horizon_ms,part.mean_normalized_gain,marker="o",label="Depth "+str(depth))
        # A zero line represents no improvement over the same-row baseline model.
        ax.axhline(0,color="black",linewidth=.8)
        # Identify both the horizon convention and the normalized metric.
        ax.set(xlabel="Selected-clock horizon (ms)",ylabel="Held-out MSE gain / training variance",title="Price-distance OBI: incremental predictive value")
        # Identify each depth curve without implying a selected winner.
        ax.legend()
        # Export a standalone figure alongside the detailed statistical tables.
        fig.tight_layout(); fig.savefig(Path(output)/"price_distance_oos_gain.png",dpi=160); plt.close(fig)

    # Compare concentration with and without location against the common OBI baseline.
    selected = summary[summary.variant.isin(["concentration","concentration_location"]) & (summary.scope=="portfolio") & (summary.geometry=="ALL") & (summary.subset=="all_l1") & (summary.horizon_ms==250)]
    # Older stores have no concentration arms and should not generate an empty chart.
    if not selected.empty:
        # Keep a small comparable plot at the prespecified primary horizon.
        fig,ax = plt.subplots(figsize=(8,5))
        # Preserve each arm's identity instead of presenting a selected winner.
        for variant,part in selected.groupby("variant"):
            # Use a stable depth order including the all-known-price window.
            part=part.set_index("depth").reindex(["3","5","10","all"])
            # The metric remains prediction loss improvement, not executable profit.
            ax.plot(range(4),part.mean_normalized_gain,marker="o",label=variant)
        # Label the windows as quoted levels rather than tick distances.
        ax.set_xticks(range(4),["3","5","10","all"])
        # Zero denotes no gain against the common same-row baseline.
        ax.axhline(0,color="black",linewidth=.8)
        # State the exact scope and metric without inferring wall honesty.
        ax.set(xlabel="Quoted depth",ylabel="Held-out MSE gain / training variance",title="250ms concentration comparisons: portfolio")
        # Keep both experimental variants visible.
        ax.legend()
        # Export a standalone figure beside the underlying metrics.
        fig.tight_layout(); fig.savefig(Path(output)/"concentration_oos_gain.png",dpi=160); plt.close(fig)

    # Compare the three new liquidity arms on the shared depth-three baseline.
    selected = summary[summary.variant.isin(["band_depth","observed_walk","attributed_flow"]) & (summary.scope=="portfolio") & (summary.geometry=="ALL") & (summary.subset=="all_l1") & (summary.depth=="3")]
    # Old feature stores legitimately lack these arms.
    if not selected.empty:
        # Keep horizon comparisons in one standalone diagnostic figure.
        fig,ax = plt.subplots(figsize=(8,5))
        # Plot every added family without hiding poor outcomes.
        for variant,part in selected.groupby("variant"):
            # Preserve temporal order rather than alphabetical horizon labels.
            part=part.sort_values("horizon_ms")
            # Gains are against the common OBI/control baseline, not against each other.
            ax.plot(part.horizon_ms,part.mean_normalized_gain,marker="o",label=variant)
        # Zero marks no incremental predictive improvement.
        ax.axhline(0,color="black",linewidth=.8)
        # Describe the actual metric without implying execution profitability.
        ax.set(xlabel="Selected-clock horizon (ms)",ylabel="Held-out MSE gain / training variance",title="Depth-three liquidity and attributed-flow comparisons")
        # Keep all experimental family names visible.
        ax.legend()
        # Export the plot beside the complete hypothesis table.
        fig.tight_layout(); fig.savefig(Path(output)/"liquidity_oos_gain.png",dpi=160); plt.close(fig)
