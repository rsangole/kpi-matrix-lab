"""Rebuild the event HTML from saved benchmark results without rerunning simulations."""
import json
from pathlib import Path
import pandas as pd
from run_suite import report

def main():
    report(pd.read_csv('data/event_suite_runs.csv'),
           pd.read_csv('data/event_suite_selected_settings.csv'),
           pd.read_csv('data/event_suite_summary.csv'),
           pd.read_csv('data/event_suite_by_scenario.csv'),
           pd.read_csv('data/event_suite_runtime.csv'),
           json.loads(Path('data/event_suite_metadata.json').read_text()))
    print('Rebuilt reports/event_suite_280.html from saved results')

if __name__=='__main__':main()
