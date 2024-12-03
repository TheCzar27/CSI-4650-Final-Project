import subprocess

def run_multiple(mode, runs=10):
    for i in range(runs):
        print(f"Running {mode} run {i + 1}/{runs}...")
        subprocess.run(["python", "src/main.py", "--mode", mode])

# Run 10 single-threaded runs
run_multiple("single", runs=10)

# Run 10 parallel runs
run_multiple("parallel", runs=10)
