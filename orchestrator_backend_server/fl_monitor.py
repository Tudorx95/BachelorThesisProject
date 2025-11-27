#!/usr/bin/env python3
"""
FL Simulation Monitor - Tool pentru monitorizarea simulărilor în timp real
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import subprocess

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_colored(text, color=WHITE):
    print(f"{color}{text}{RESET}")

def clear_screen():
    subprocess.run(['clear'], shell=True)

def read_log_file(log_path, last_n_lines=50):
    """Read last N lines from log file"""
    if not log_path.exists():
        return None
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            return ''.join(lines[-last_n_lines:])
    except Exception as e:
        return f"Error reading log: {str(e)}"

def monitor_simulation(base_dir, user_id, task_id, refresh_interval=5):
    """Monitor a running simulation"""
    
    simulation_dir = Path(base_dir) / f"user_{user_id}" / task_id
    log_dir = simulation_dir / "logs"
    results_dir = simulation_dir / "results"
    
    print_colored(f"\n{BOLD}FL SIMULATION MONITOR", CYAN)
    print_colored("="*60, CYAN)
    print(f"Task ID: {task_id}")
    print(f"User ID: {user_id}")
    print(f"Directory: {simulation_dir}")
    print_colored("="*60, CYAN)
    
    if not simulation_dir.exists():
        print_colored(f"✗ Simulation directory not found!", RED)
        return
    
    # Log files to monitor
    log_files = {
        'Main Log': log_dir / 'simulation.log',
        'Template': log_dir / 'template_execution.log',
        'Download Data': log_dir / 'download_data.log',
        'Poison Data': log_dir / 'poison_data.log',
        'FL Clean': log_dir / 'fl_simulation_clean.log',
        'FL Poisoned': log_dir / 'fl_simulation_poisoned.log'
    }
    
    # Result files to check
    result_files = {
        'Config': simulation_dir / 'simulation_config.json',
        'Clean Results': results_dir / 'fl_clean.json',
        'Poisoned Results': results_dir / 'fl_poisoned.json',
        'Analysis': results_dir / 'analysis.json',
        'Summary': results_dir / 'summary.txt'
    }
    
    try:
        while True:
            clear_screen()
            
            # Header
            print_colored(f"\n{BOLD}FL SIMULATION MONITOR", CYAN)
            print_colored("="*60, CYAN)
            print(f"Task ID: {task_id}")
            print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            print_colored("="*60, CYAN)
            
            # Check configuration
            if result_files['Config'].exists():
                with open(result_files['Config']) as f:
                    config = json.load(f)
                print_colored("\nConfiguration:", YELLOW)
                print(f"  Clients: {config.get('N', '?')} (Malicious: {config.get('M', '?')})")
                print(f"  Rounds: {config.get('ROUNDS', '?')}")
                print(f"  Strategy: {config.get('strategy', '?')}")
                print(f"  Poison: {config.get('poison_operation', '?')} "
                      f"(intensity: {config.get('poison_intensity', '?')}, "
                      f"percentage: {config.get('poison_percentage', '?')})")
            
            # Status from log files
            print_colored("\n📊 Pipeline Status:", YELLOW)
            for name, log_path in log_files.items():
                if log_path.exists():
                    size = log_path.stat().st_size
                    modified = datetime.fromtimestamp(log_path.stat().st_mtime)
                    age = (datetime.now() - modified).total_seconds()
                    
                    if age < 60:  # Active in last minute
                        status = f"{GREEN}✓ Active{RESET}"
                    elif size > 0:
                        status = f"{BLUE}✓ Completed{RESET}"
                    else:
                        status = f"{YELLOW}⚠ Empty{RESET}"
                    
                    print(f"  {name:15} : {status} (Size: {size:,} bytes)")
                else:
                    print(f"  {name:15} : {YELLOW}⏳ Waiting...{RESET}")
            
            # Check results
            print_colored("\n📈 Results:", YELLOW)
            
            # Clean results
            if result_files['Clean Results'].exists():
                try:
                    with open(result_files['Clean Results']) as f:
                        clean_data = json.load(f)
                    rounds = len(clean_data.get('round_metrics_history', []))
                    accuracy = clean_data.get('final_accuracy', 0)
                    
                    if rounds > 0:
                        print(f"  Clean:     {GREEN}✓{RESET} Accuracy: {accuracy:.4f} ({rounds} rounds)")
                    else:
                        print(f"  Clean:     {YELLOW}⚠{RESET} No training history (default file)")
                except:
                    print(f"  Clean:     {RED}✗{RESET} Error reading results")
            else:
                print(f"  Clean:     {YELLOW}⏳{RESET} In progress...")
            
            # Poisoned results
            if result_files['Poisoned Results'].exists():
                try:
                    with open(result_files['Poisoned Results']) as f:
                        poisoned_data = json.load(f)
                    rounds = len(poisoned_data.get('round_metrics_history', []))
                    accuracy = poisoned_data.get('final_accuracy', 0)
                    
                    if rounds > 0:
                        print(f"  Poisoned:  {GREEN}✓{RESET} Accuracy: {accuracy:.4f} ({rounds} rounds)")
                    else:
                        print(f"  Poisoned:  {YELLOW}⚠{RESET} No training history")
                except:
                    print(f"  Poisoned:  {RED}✗{RESET} Error reading results")
            else:
                print(f"  Poisoned:  {YELLOW}⏳{RESET} Waiting...")
            
            # Final analysis
            if result_files['Analysis'].exists():
                try:
                    with open(result_files['Analysis']) as f:
                        analysis = json.load(f)
                    
                    print_colored("\n✅ SIMULATION COMPLETE!", GREEN)
                    print(f"  Clean Accuracy:    {analysis.get('clean_accuracy', 0):.4f}")
                    print(f"  Poisoned Accuracy: {analysis.get('poisoned_accuracy', 0):.4f}")
                    print(f"  Accuracy Drop:     {analysis.get('accuracy_drop', 0):.4f}")
                    
                    # Show summary
                    if result_files['Summary'].exists():
                        print_colored("\n📝 Summary:", CYAN)
                        with open(result_files['Summary']) as f:
                            print(f.read())
                    
                    print_colored("\n✨ Pipeline completed successfully!", GREEN)
                    break
                except:
                    pass
            
            # Show recent logs
            main_log = log_dir / 'simulation.log'
            if main_log.exists():
                print_colored("\n📜 Recent Activity (Main Log):", YELLOW)
                recent = read_log_file(main_log, 10)
                if recent:
                    for line in recent.split('\n')[-10:]:
                        if line:
                            if 'ERROR' in line or 'failed' in line.lower():
                                print_colored(f"  {line}", RED)
                            elif 'SUCCESS' in line or '✓' in line:
                                print_colored(f"  {line}", GREEN)
                            elif 'WARNING' in line:
                                print_colored(f"  {line}", YELLOW)
                            else:
                                print(f"  {line}")
            
            # Check for active FL simulation logs
            for log_name in ['fl_simulation_clean.log', 'fl_simulation_poisoned.log']:
                log_path = log_dir / log_name
                if log_path.exists():
                    # Check if file is being written to
                    current_size = log_path.stat().st_size
                    time.sleep(1)
                    if log_path.stat().st_size > current_size:
                        # File is growing - simulation is running
                        print_colored(f"\n🔄 Active: {log_name}", GREEN)
                        recent = read_log_file(log_path, 5)
                        if recent:
                            last_lines = recent.split('\n')[-5:]
                            for line in last_lines:
                                if 'Round' in line or 'Epoch' in line:
                                    print_colored(f"  {line}", CYAN)
            
            print(f"\n{YELLOW}Refreshing in {refresh_interval} seconds... (Ctrl+C to exit){RESET}")
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print_colored("\n\n👋 Monitoring stopped by user", YELLOW)
        return

def check_simulation_status(base_dir, user_id, task_id):
    """Quick check of simulation status"""
    simulation_dir = Path(base_dir) / f"user_{user_id}" / task_id
    
    if not simulation_dir.exists():
        print_colored(f"✗ Simulation not found: {task_id}", RED)
        return
    
    print_colored(f"\n📊 Simulation Status: {task_id}", CYAN)
    print("="*50)
    
    # Check key files
    checks = {
        'Template': simulation_dir / 'template_code.py',
        'Config': simulation_dir / 'simulation_config.json',
        'Main Log': simulation_dir / 'logs' / 'simulation.log',
        'Clean Results': simulation_dir / 'results' / 'fl_clean.json',
        'Poisoned Results': simulation_dir / 'results' / 'fl_poisoned.json',
        'Analysis': simulation_dir / 'results' / 'analysis.json'
    }
    
    for name, path in checks.items():
        if path.exists():
            size = path.stat().st_size
            if size > 0:
                print(f"✓ {name:20} : {size:,} bytes")
            else:
                print(f"⚠ {name:20} : Empty file")
        else:
            print(f"✗ {name:20} : Not found")
    
    # Check if completed
    analysis_path = simulation_dir / 'results' / 'analysis.json'
    if analysis_path.exists():
        try:
            with open(analysis_path) as f:
                analysis = json.load(f)
            
            print_colored("\n✅ Simulation Complete!", GREEN)
            print(f"Clean Accuracy: {analysis.get('clean_accuracy', 0):.4f}")
            print(f"Poisoned Accuracy: {analysis.get('poisoned_accuracy', 0):.4f}")
            print(f"Accuracy Drop: {analysis.get('accuracy_drop', 0):.4f}")
        except:
            print_colored("\n⚠ Analysis file exists but may be corrupted", YELLOW)
    else:
        print_colored("\n⏳ Simulation in progress or not started", YELLOW)

def tail_log(base_dir, user_id, task_id, log_type='main'):
    """Tail a specific log file"""
    simulation_dir = Path(base_dir) / f"user_{user_id}" / task_id
    log_dir = simulation_dir / "logs"
    
    log_files = {
        'main': 'simulation.log',
        'template': 'template_execution.log',
        'download': 'download_data.log',
        'poison': 'poison_data.log',
        'clean': 'fl_simulation_clean.log',
        'poisoned': 'fl_simulation_poisoned.log'
    }
    
    if log_type not in log_files:
        print_colored(f"✗ Unknown log type: {log_type}", RED)
        print(f"Available: {', '.join(log_files.keys())}")
        return
    
    log_path = log_dir / log_files[log_type]
    
    if not log_path.exists():
        print_colored(f"✗ Log file not found: {log_path}", RED)
        return
    
    print_colored(f"📜 Tailing {log_type} log for {task_id}", CYAN)
    print("="*50)
    
    # Use subprocess to tail the file
    try:
        subprocess.run(['tail', '-f', str(log_path)])
    except KeyboardInterrupt:
        print_colored("\n👋 Stopped tailing", YELLOW)

def main():
    parser = argparse.ArgumentParser(description='FL Simulation Monitor')
    parser.add_argument('--base-dir', default='/home/tudor.lepadatu/Licenta/Part2/fl_simulations',
                        help='Base directory for simulations')
    parser.add_argument('--user', '-u', default='1', help='User ID')
    parser.add_argument('--task', '-t', required=True, help='Task ID')
    parser.add_argument('--mode', '-m', choices=['monitor', 'status', 'tail'], 
                        default='monitor', help='Monitoring mode')
    parser.add_argument('--log', '-l', default='main',
                        choices=['main', 'template', 'download', 'poison', 'clean', 'poisoned'],
                        help='Log type to tail (for tail mode)')
    parser.add_argument('--refresh', '-r', type=int, default=5,
                        help='Refresh interval in seconds (for monitor mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'monitor':
        monitor_simulation(args.base_dir, args.user, args.task, args.refresh)
    elif args.mode == 'status':
        check_simulation_status(args.base_dir, args.user, args.task)
    elif args.mode == 'tail':
        tail_log(args.base_dir, args.user, args.task, args.log)

if __name__ == '__main__':
    main()