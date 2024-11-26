# import concurrent.futures
import multiprocessing
import time
import random
# import numpy as np
# import datetime as dt
# from qs_mps.mps_class import MPS
# from qs_mps. utils import tensor_shapes, get_precision


def run_with_timeout(func, args, timeout):
    with multiprocessing.Pool() as pool:
        result = pool.apply_async(func, args)
        try:
            return result.get(timeout=timeout)
        except multiprocessing.TimeoutError:
            print(f"\n## TimeoutError: Algorithm exceeded {timeout} seconds ##\n")
            pool.terminate()
            # pool.join()  # Forcefully terminate the worker
            # result = pool.apply_async(func, default_args)
            # pool.join()
            # return None  # Return None or any indicator of failure
            return None


def dummy(n, params):
    print(f"trying to sleep for {n} seconds")
    print(f"params = {params}")
    time.sleep(n)
    print(f"slept for {n} seconds")
    return n


nums = [ random.randint(0, 5) for _ in range(10) ]
print(f"nums = {nums}")
timeout = 2.1

args = dict(guess="pippo", porc="dio")
start = time.perf_counter()
for num in nums:
    # var = 0
    for attempt in range(3):
        print(f"\n*** Attempt n. {attempt + 1}, num = {num} ***\n")
        params = (args, 0)
        result = run_with_timeout(dummy, (num, params), timeout=timeout)
        if result is None:
            print("processed timeout, trying again")
            args["guess"] = "pappo"
            # result = run_with_timeout(dummy, (2, params), timeout=timeout)
            num = 2
        else:
            print("not failed")
            args["guess"] = "pippo"
            print(f"  result = {result}")
            break
    print(f"  args = {args}, params = {params}")
stop = time.perf_counter()

trunc_nums = [ n if n < timeout else timeout + 2 for n in nums ]

print(f"nums       = {nums}")
print(f"trunc_nums = {trunc_nums}")
print(f"Total nums = {sum(nums)}")
print(f"Total truncated nums = {sum(trunc_nums)}")
print(f"Elapsed time: {stop - start}")
