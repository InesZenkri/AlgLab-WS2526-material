



my first approech was fine for the first 4 instances
but it got a timeout on the 5th one:
Progress:  80%|███████████████████████████████████▏        | 4/5 [01:48<00:27, 27.22s/it]
Traceback (most recent call last):
  File "C:\Users\inezen\Desktop\AlgLab-WS2526-material\sheets\03_sat\exercises\01_k_centers\_alglab_utils.py", line 75, in run_in_subprocess
    stdout, _ = proc.communicate(timeout=self.max_runtime_s)
                ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python313\Lib\subprocess.py", line 1240, in communicate
    sts = self.wait(timeout=self._remaining_time(endtime))
  File "C:\Program Files\Python313\Lib\subprocess.py", line 1276, in wait
    return self._wait(timeout=timeout)
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^

i was rebuilding the solver each time in the while loop, so i took it out, and now build once and reuse 