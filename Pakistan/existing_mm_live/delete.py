import re
import pandas as pd
from pathlib import Path


def categorize_file(filename, docstring):
    filename = filename.lower()

    # 1. Sweeps & Grid Searches
    if any(x in filename for x in ['sweep', 'horserace']):
        return 'Sweep / Experiment'
    # 2. Diagnostics, Sanity Checks, and Validations
    elif any(x in filename for x in ['diag_', 'test_', 'check_', 'verify_', 'diagnose_', 'sanity']):
        return 'Diagnostic / Analysis'
    # 3. Parameter Calibrations & Profile Builders
    elif any(x in filename for x in ['calibrate_', 'profile_', 'build_watchlist']):
        return 'Calibration / Pre-computation'
    # 4. Data Extraction, Snapshot Prep, and Cleaning
    elif any(x in filename for x in ['feature_store', 'snapshot_prep', 'corp_action', 'parser', 'fills_to_csv']):
        return 'Data Pipeline / Extractor'
    # 5. Post-Trade Analytics, P&L, and Markout
    elif any(x in filename for x in ['attribution', 'pnl', 'markout', 'kyle_lambda', 'score']):
        return 'Analytics / Attribution'
    # 6. Futures-Specific Logic
    elif 'fut_' in filename or 'futures' in filename:
        return 'Futures Market Making'
    # 7. Core Strategy Engines
    elif 'micro_mm' in filename:
        return 'Live Quoting Engine / Strategy'
    # 8. Core Backtesting Engines
    elif filename.startswith('mm_backtest'):
        return 'Backtest Engine'
    # 9. Harnesses & Batch Runners
    elif filename in ['run_legacy_mm.py', 'run_mm_batch.py', 'run_naive_two.py', 'mm_harness.py', 'universe_run.py',
                      'run_all_tickers.py', 'run_daily_stats.py']:
        return 'Runner / Harness'
    # 10. Global Configs & State Trackers
    elif filename in ['config.py', 'halt_state.py', 'ticker_stats_core.py']:
        return 'Configuration / Core Scaffolding'
    else:
        return 'Utility / Support'


def extract_important_info(docstring):
    """Pulls critical architectural warnings, hypotheses, or flagged simplifications."""
    keywords = ['FLAGGED SIMPLIFICATION', 'WARNING', 'NOTE', 'WHY', 'HYPOTHESIS', 'THE EXPERIMENT']
    found = []
    lines = docstring.split('\n')
    for line in lines:
        if any(k in line.upper() for k in keywords):
            # Clean and truncate to avoid massive Excel cells
            cleaned = line.strip().lstrip('#').strip()
            found.append(cleaned[:150] + ("..." if len(cleaned) > 150 else ""))
    return " | ".join(found) if found else "None"


def parse_inventory(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split the document by the file separator generated in your bash loop
    parts = re.split(r'=====FILE:\s*(.*?)\s*=====\n', content)[1:]

    data = []
    for i in range(0, len(parts), 2):
        filename = parts[i].strip()
        body = parts[i + 1]

        # Split into Docstring, Imports, and Outputs
        doc_split = body.split('--- imports ---')
        docstring = doc_split[0].replace('--- docstring/top ---', '').strip()

        if len(doc_split) > 1:
            imp_out_split = doc_split[1].split('--- outputs (csv/parquet writes) ---')
            imports = imp_out_split[0].strip().replace('\n', ', ')
            outputs = imp_out_split[1].strip().replace('\n', ', ') if len(imp_out_split) > 1 else "None"
        else:
            imports, outputs = "None", "None"

        # Clean up the docstring to generate a 1-2 sentence summary
        lines = [line.strip().lstrip('"""').lstrip("'''").lstrip('#').strip() for line in docstring.split('\n')]
        lines = [line for line in lines if line]
        summary = " ".join(lines[:3])
        if len(summary) > 200:
            summary = summary[:197] + "..."

        category = categorize_file(filename, docstring)
        important_info = extract_important_info(docstring)

        data.append({
            'File Name': filename.replace('./', ''),
            'What it does / tests': summary if summary else "No docstring provided.",
            'Imports': imports if imports else "None",
            'Outputs': outputs if outputs else "None",
            'Category': category,
            'Important Info Missed': important_info
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    input_file = '/tmp/code_inventory.txt'
    output_file = '/Users/shazzak/PycharmProjects/HFT/existing_mm_live/hft_file_inventory.xlsx'

    df = parse_inventory(input_file)
    df.to_excel(output_file, index=False)
    print(f"Successfully categorized {len(df)} files.")
    print(f"Excel file generated at: {output_file}")