import psutil
PROCNAME = "acquire.py"

for proc in psutil.process_iter():
    # check whether the process name matches
    #if proc.name() == PROCNAME:
    #    proc.kill()
    # check whether the process name matches
    cmdline = proc.cmdline()
    if len(cmdline) >= 2 and "python" in cmdline[0] and cmdline[1] == PROCNAME:
        # print(proc)
        proc.kill()

