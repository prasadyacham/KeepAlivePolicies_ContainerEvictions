import numpy as np
import random
from collections import defaultdict
from LambdaData import *
from Container import *
import os


class LambdaScheduler:

    def __init__(self, policy: str = "RAND", mem_capacity: int = 32000, num_funcs: int = 10, run: str = "a",
                 log_dir=""):
        fname = "{}-{}-{}-{}-".format(policy, num_funcs, mem_capacity, run)

        self.mem_capacity = mem_capacity
        self.mem_used = 0
        self.eviction_policy = policy

        self.wall_time = 0  # Current system time
        # Container : (launch_time, launch_time+processing_time)
        self.RunningC = dict()
        self.ContainerPool = []  # simple list of `Container`s
        # list of tuplies (`LambdaData`, invocation_time)
        self.FunctionHistoryList = []

        self.PerfLogFName = os.path.join(log_dir, fname + "performancelog.csv")
        self.PerformanceLog = open(self.PerfLogFName, "w")
        self.PerformanceLog.write("lambda,time,meta\n")

        self.evdict = defaultdict(int)
        self.capacity_misses = defaultdict(int)

        self.provider_overhead_base = 3000  # 3 seconds
        self.provider_overhead_pct = 0.2  # 20% of function runtime added to cold start

        self.Clock = 0
        self.FunctionFreq = dict()

        # Multiple Policies Created
        if self.eviction_policy == "RAND":
            # Function to be called pick containers to evict
            self.EvictionFunc = self.RandomEvictionPicker
        elif self.eviction_policy == "LEAST_USED":
            # Function to be called pick containers to evict
            self.EvictionFunc = self.LeastUsedEvictionPicker
        elif self.eviction_policy == "MAX_MEM":
            # Function to be called pick containers to evict
            self.EvictionFunc = self.MaxMemoryEvictionPicker
        elif self.eviction_policy == "CLOUD21":
            # Function to be called pick containers to evict
            self.EvictionFunc = self.GreedyDualEvictionPicker
        else:
            raise NotImplementedError(
                "Unkonwn eviction policy: {}".format(self.eviction_policy))

    ##############################################################

    def WritePerfLog(self, d: LambdaData, time, meta):
        msg = "{},{},{}\n".format(d.kind, time, meta)
        self.PerformanceLog.write(msg)

    ##############################################################

    def AssertMemory(self):
        """ Raise an exception if the memory assumptions of the simulation have been violated """
        used_mem = sum([c.metadata.mem_size for c in self.ContainerPool])
        if used_mem != self.mem_used:
            raise Exception("Container pool mem '{}' does not match tracked usage '{}'".format(
                used_mem, self.mem_used))
        if used_mem > self.mem_capacity:
            raise Exception("Container pool mem '{}' exceeds capacity '{}'".format(
                used_mem, self.mem_capacity))

    ##############################################################

    def ColdHitProcTime(self, d: LambdaData) -> float:
        """
        Total processing time for a cold hit on the given lambda
        """
        return self.provider_overhead_base + d.run_time + (self.provider_overhead_pct * d.run_time)

    ##############################################################

    def find_container(self, d: LambdaData):
        """ 
        Search through the containerpool for a non-running container with the sane metadata as `d`
        Return None if one cannot be found
        """
        if len(self.ContainerPool) == 0:
            return None
        containers_for_the_lambda = [x for x in self.ContainerPool if (x.metadata == d and
                                                                       x not in self.RunningC)]

        if containers_for_the_lambda == []:
            return None
        else:
            return containers_for_the_lambda[0]
        # Just return the first element.

    ##############################################################

    def container_clones(self, c: Container):
        """ Return all the conatienrs have the same function data as `c` """
        return [x for x in self.ContainerPool if x.metadata == c.metadata]

    ##############################################################

    def CheckFree(self, c):
        """
        Check
        """
        mem_size = c.metadata.mem_size
        return mem_size + self.mem_used <= self.mem_capacity

    ##############################################################

    def AddToPool(self, c: Container):
        """ Add contaienr to the ContainerPool, maintaining bookkeeping """
        mem_size = c.metadata.mem_size
        if mem_size + self.mem_used <= self.mem_capacity:
            # Have free space
            self.mem_used = self.mem_used + mem_size
            c.clock = self.Clock
            c.priority = self.get_priority(c)
            self.ContainerPool.append(c)
            return True
        else:
            # print ("Not enough space for memsize, used, capacity.", mem_size, self.mem_used, self.mem_capacity)
            return False

    ##############################################################

    def RemoveFromPool(self, c: Container):
        if c in self.RunningC:
            raise Exception("Cannot remove a running container")
        self.ContainerPool.remove(c)
        self.mem_used -= c.metadata.mem_size

    ############################################################

    def RandomEvictionPicker(self, to_free):
        """ 
        Return victim lists
        Simple eviction that randomly chooses from non-running containers
        """
        eviction_list = []
        # XXX Can't evict running containers!
        # Even with infinite concurrency, container will still exist in running_c
        available = [c for c in self.ContainerPool if c not in self.RunningC]

        while to_free > 0 and len(available) > 0:
            victim = random.choice(available)
            available.remove(victim)
            eviction_list.append(victim)
            to_free -= victim.metadata.mem_size

        return eviction_list

    ############################################################

    def LeastUsedEvictionPicker(self, to_free):
        """ 
        Return victim lists
        Simple eviction that chooses least used function from non-running containers
        """
        eviction_list = []
        # XXX Can't evict running containers!
        # Even with infinite concurrency, container will still exist in running_c
        available = [c for c in self.ContainerPool if c not in self.RunningC]
        sorted_available = sorted(available,
                                  key=lambda c: self.FunctionFreq.get(c.metadata.kind, 0))

        while to_free > 0 and len(sorted_available) > 0:
            victim = sorted_available[0]
            sorted_available.remove(victim)
            eviction_list.append(victim)
            to_free -= victim.metadata.mem_size

        return eviction_list

    #############################################################

    def MaxMemoryEvictionPicker(self, to_free):
        """ 
        Return victim lists
        Simple eviction that chooses least used function from non-running containers
        """
        eviction_list = []
        # XXX Can't evict running containers!
        # Even with infinite concurrency, container will still exist in running_c
        available = [c for c in self.ContainerPool if c not in self.RunningC]
        sorted_available = sorted(available, key=lambda c: c.metadata.mem_size, reverse=True)

        while to_free > 0 and len(sorted_available) > 0:
            victim = sorted_available[0]
            sorted_available.remove(victim)
            eviction_list.append(victim)
            to_free -= victim.metadata.mem_size

        return eviction_list

    #############################################################

    def GreedyDualEvictionPicker(self, to_free):
        """ 
        Return victim lists
        Simple eviction that chooses least used function from non-running containers
        """
        eviction_list = []
        # XXX Can't evict running containers!
        # Even with infinite concurrency, container will still exist in running_c
        available = [c for c in self.ContainerPool if c not in self.RunningC]
        sorted_available = sorted(available, key=lambda c: c.priority)

        # Added logic on Greedy dual policy to get rid of containers which are duplicate.
        # With only greedy dual the performance was not as expected
        seen = []
        available_dup_func_c_list = []
        for c in sorted_available:
            if c.metadata.kind in seen:
                available_dup_func_c_list.append(c)
            else:
                seen.append(c.metadata.kind)

        while to_free > 0 and len(sorted_available) > 0:
            if len(available_dup_func_c_list) > 0:
                victim = available_dup_func_c_list[0]
                available_dup_func_c_list.remove(victim)
            else:
                victim = sorted_available[0]
            # victim = sorted_available[0]
            sorted_available.remove(victim)
            eviction_list.append(victim)
            to_free -= victim.metadata.mem_size

        if len(eviction_list) > 0:
            self.Clock = eviction_list[-1].priority  # Last item will be least priority

        return eviction_list

    #############################################################
    # Function to calculate priority for each container
    # Idea inherited from https://cgi.luddy.indiana.edu/~prateeks/papers/faascache-asplos21.pdf but added extra
    # logic in Greedy dual eviction picker to get rid of duplicate containers with least priority instead of only
    # least priority

    def get_priority(self, c: Container):
        freq = self.FunctionFreq.get(c.metadata.kind, 0)
        cost = float(c.metadata.run_time - c.metadata.warm_time)
        size = c.metadata.mem_size
        priority = c.clock + freq * (cost / size)
        return priority

    #############################################################

    def Eviction(self, d: LambdaData):
        """ Return a list of containers that have been evicted """
        if len(self.RunningC) == len(self.ContainerPool):
            # all containers busy
            return []

        eviction_list = self.EvictionFunc(to_free=d.mem_size)

        for v in eviction_list:
            self.RemoveFromPool(v)
            # self.mem_used -= v.metadata.mem_size
            k = v.metadata.kind
            self.evdict[k] += 1

        return eviction_list

    ##############################################################

    def cache_miss(self, d: LambdaData):
        """ 
        A cache miss for the function.
        Create a new Container that has been added to the Container Pool and return it
        Return None if one could not be created

        Evicts non-running containers in an attempt to make room
        """
        c = Container(d)
        if not self.CheckFree(c):  # due to space constraints
            # Is a list. containers already terminated
            evicted = self.Eviction(d)

        added = self.AddToPool(c)
        if not added:
            # unable to add a new container due to memory constraints
            return None

        return c

    ##############################################################

    def cleanup_finished(self):
        """ Go through running containers, remove those that have finished """
        t = self.wall_time
        finished = []
        for c in self.RunningC:
            (start_t, fin_t) = self.RunningC[c]
            if t >= fin_t:
                finished.append(c)

        for c in finished:
            del self.RunningC[c]

        return len(finished)

    ##############################################################

    def runInvocation(self, d: LambdaData, t=0):
        """ Entrypoint for the simulation """
        self.wall_time = t
        self.cleanup_finished()

        func_kind = d.kind
        self.FunctionFreq[func_kind] = self.FunctionFreq.get(func_kind, 0) + 1

        c = self.find_container(d)
        if c is None:
            # Launch a new container since we didnt find one for the metadata ...
            c = self.cache_miss(d)
            if c is None:
                # insufficient memory
                self.capacity_misses[d.kind] += 1
                return
            c.run()
            processing_time = self.ColdHitProcTime(d)
            self.RunningC[c] = (t, t + processing_time)
            self.WritePerfLog(d, t, "miss")

        else:
            c.clock = self.Clock
            c.priority = self.get_priority(c)
            c.run()
            processing_time = d.warm_time
            self.RunningC[c] = (t, t + processing_time)
            self.WritePerfLog(d, t, "hit")

        # Updating the priority of all containers
        for x in self.ContainerPool:
            if x.metadata == c.metadata:
                x.priority = c.priority
        self.FunctionHistoryList.append((d, t))
        self.AssertMemory()

    ##############################################################

    def miss_stats(self):
        """ Go through the performance log."""
        rdict = dict()  # For each activation
        with open(self.PerfLogFName, "r") as f:
            line = f.readline()  # throw away header
            for line in f:
                line = line.rstrip()
                d, ptime, evtype = line.split(",")
                k = d
                if k not in rdict:
                    mdict = dict()
                    mdict['misses'] = 0
                    mdict['hits'] = 0
                    rdict[k] = mdict

                if evtype == "miss":
                    rdict[k]['misses'] = rdict[k]['misses'] + 1
                elif evtype == "hit":
                    rdict[k]['hits'] = rdict[k]['hits'] + 1
                else:
                    pass

        # Also some kind of response time data?
        return rdict

    ##############################################################
    ##############################################################
    ##############################################################


if __name__ == "__main__":
    from pprint import pprint
    import pickle

    ls = LambdaScheduler(policy="RAND", mem_capacity=2048,
                         num_funcs=20, run="b")

    pth = "../../traces/20-b.pckl"
    with open(pth, "r+b") as f:
        lambdas, input_trace = pickle.load(f)
    print(len(input_trace))

    for d, t in input_trace:
        ls.runInvocation(d, t)

    print("\n\nDONE\n")

    pprint(ls.evdict)
    pprint(ls.miss_stats())
    print("cap", ls.capacity_misses)
