import sys, random, math
from twisted.spread import pb
from twisted.internet import reactor, defer, task
import datetime
import time




# class solver():
#     def __init__(self):
#         pass
#         # print 'solver, solver, solver ', self.id

    
#     # Alias methods, for demonstration version:  we use aliases with prefix remote, because subprocess understand a remote method with prefixed remote
    # remote_step1 = scheduled_jobs_process1
    # remote_step2 = scheduled_jobs_process2   # step2




class Solver(pb.Root):

    def __init__(self, id):
        print 'ttttttttttttttttttttttttttt ', id
        print "solver.py %s: solver init" % id
        self.id = id

    def __str__(self): # String representation
        return "Solver %s" % self.id

    def remote_initialize(self, initArg):
        return "%s initialized" % self

    def remote_status(self):
        print "solver.py %s: remote_status" % self.id
        return "%s operational" % self

    def remote_terminate(self):
        print "solver.py %s: remote_terminate" % self.id
        reactor.callLater(0.5, reactor.stop)
        return "%s terminating..." % self

    def sleep(self, secs):
        d = defer.Deferred()
        reactor.callLater(secs, d.callback, None)
        return d

    def scheduled_jobs_process1(self, arg):
        print 'server_solver.py: Initiating scheduled_jobs_process1 ........................................'
        l1 = task.LoopingCall(getattr(self, "step1"), arg)
        l1.start(0.0001)

    @defer.inlineCallbacks
    def step1(self, arg):
        print "solver.py %s: solver step" % self.id
        curr_time= time.strptime(str(datetime.datetime.now().strftime("%H:%M:%S")), "%H:%M:%S")
        curr_time_sec = datetime.timedelta(hours=curr_time.tm_hour, minutes=curr_time.tm_min, seconds=curr_time.tm_sec).seconds
        print curr_time_sec
        print "Waiting for 10 seconds"
        yield self.sleep(10)
        # print 'solver.py %s: solver did its job' %self.id
        # print "Simulate work and return result"
        # result = 0
        # for i in range(random.randint(1000000, 3000000)):
        #     angle = math.radians(random.randint(0, 45))
        #     result += math.tanh(angle)/math.cosh(angle)
        print 'solver.py %s: solver did its job' %self.id
        yield "%s, %s" % (self, 'lalalalalalalalalalalalalalala')
        # yield "%s, %s, result: %.2f" % (self, str(arg), result)

    def scheduled_jobs_process2(self, arg):
        print 'server_solver.py: Initiating scheduled_jobs_process2'
        l2 = task.LoopingCall(getattr(self, "step2"), arg)
        l2.start(0.0001)


    @defer.inlineCallbacks
    def step2(self, arg):
        print "solver.py %s: solver step" % self.id
        # print "Simulate work and return result"
        curr_time= time.strptime(str(datetime.datetime.now().strftime("%H:%M:%S")), "%H:%M:%S")
        curr_time_sec = datetime.timedelta(hours=curr_time.tm_hour, minutes=curr_time.tm_min, seconds=curr_time.tm_sec).seconds
        print curr_time_sec
        print "Waiting for 2 seconds"
        yield self.sleep(2)
        print 'solver.py %s: solver did its job' %self.id
        yield "%s, %s" % (self, 'print print print print print')


    remote_step1 = scheduled_jobs_process1
    remote_step2 = scheduled_jobs_process2





if __name__ == "__main__":
    print 'Inside the __name__'
    print sys.argv[1]
    port = int(sys.argv[1])
    print 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee ', Solver(sys.argv[1])
    print 'rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr' , pb.PBServerFactory(Solver(sys.argv[1]))
    reactor.listenTCP(port, pb.PBServerFactory(Solver(sys.argv[1])))
    print 'Created the service created the service'
    reactor.run()