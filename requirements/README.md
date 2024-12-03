# CSI-4650 Final Project
 Final project demonstrating parallel computing efficiency over single-threaded computing.

 Requirements:
 pip install torch torchvision matplotlib numpy pillow 
 pip-24.3.1 
 pip install pandas

 To Reproduce the results as seen in the graph below:
 ![alt text](image.png)

 Begin by opening the bash terminal and utilize the multi-run file to execute 10 runs each of single threaded
 and parallel runs. Use the command: python src/multi_run.py

Once the runs are completed, we can run another command to see a bar chart of the mean of the single-threaded and
parallel runs. Use the command: python benchmarks/performance_plots.py

This image can be saved into the program.

Finally, we must identify the hardware being used so we can justify the program. We can use the executable included
in this program to identify the CPU, cores, threads, RAM, OS, OS Version and Python version. To do this use the command: python src/hardware_info.py

If only one run is needed of the single-threaded or parallel image processing, you can use the commands below to achieve either:
Single-threaded run: python src/main.py --mode single
Parallel run: python src/main.py --mode parallel