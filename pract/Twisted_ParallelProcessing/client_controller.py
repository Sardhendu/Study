import sys
from subprocess import Popen
from twisted.spread import pb
from twisted.internet import reactor, defer

START_PORT = 8800#8686#5566
MAX_PROCESSES = 2

class Controller(object):
   
    def checkStatus(self):
        print "controller.py: checkStatus"
        for solver in self.solvers.values():
            solver.callRemote("status").addCallbacks(
                lambda r: sys.stdout.write(r + "\n"), self.failed, 
                errbackArgs=("Status Check Failed"))
                                                     
    def failed(self, results, failureMessage="Call Failed"):
        print "controller.py: failed"
        for (success, returnValue), (address, port) in zip(results, self.solvers):
            if not success:
                raise Exception("address: %s port: %d %s" % (address, port, failureMessage))

    def __init__(self):
        print "controller.py: init"
        self.solvers = dict.fromkeys([("127.0.0.1", i) for i in range(START_PORT, START_PORT+MAX_PROCESSES)])
        print 'The solvers are: ', self.solvers 
        # self.pids = [Popen(["python", "solver.py", str(port)]).pid for ip, port in self.solvers]   # This command will run the solver file
        self.pids = [] 
        for ip, port in self.solvers:
            a = Popen(["python", "server_solver.py", str(port)]).pid
            self.pids.append(a)
        print "PIDS: ", self.pids
        self.connected = False
        reactor.callLater(5, self.connect) # Here we delay the client request to connect, so that the server starts first. A five second delay would be okay, for safety it is okay to write 10 seconds, for slow machines

    def storeConnections(self, results):
        print "controller.py: storeconnections"
        for (success, solver), (address, port) in zip(results, self.solvers):
            self.solvers[address, port] = solver
        print "controller.py: Connected; self.solvers:", self.solvers
        self.connected = True


    def connect(self):
        print "controller.py: connect"
        connections = []
        for address, port in self.solvers:
            print 'The address and port are: ', address, port
            factory = pb.PBClientFactory()
            reactor.connectTCP(address, port, factory)
            connections.append(factory.getRootObject())
        defer.DeferredList(connections, consumeErrors=True).addCallbacks(
            self.storeConnections, self.failed, errbackArgs=("Failed to Connect"))

        print "controller.py: starting parallel jobs"
        self.start()


    def broadcast_parallel_jobs(self, processes, jobs):
        print 'controller.py: Broadcasting Parallel jobs'
        deferreds = []
        if len(processes) == len(jobs):
            for process, job in zip(processes, jobs):
                remoteMethodName = job[0]
                arguments = job[1]
                print remoteMethodName
                print arguments
                deferreds.append(process.callRemote(remoteMethodName, arguments))
        else:
            print "There are more parallel jobs than processes"
        print deferreds
        # reactor.callLater(3, self.checkStatus)
        # defer.DeferredList(deferreds, consumeErrors=True).addCallbacks(
        #     self.collectResults, self.failed)    #  errbackArgs=(failureMessage)


    def start(self):
        print "controller.py: Begin the solving process"
        if not self.connected:
            print 'Not connected, trying again'
            return reactor.callLater(0.5, self.start)
        print 'Connected, Moving ahead'
        job_1 = ["step1", ("step 1"), None, "Failed Step 1"]
        job_2 = ["step2", ("step 2"), None, "Failed Step 2"]
        job_list = [job_1, job_2]
        processes = [solver for solver in self.solvers.values()]
        self.broadcast_parallel_jobs(processes=processes, jobs=job_list)

    # def collectResults(self, results):
    #     print "controller.py: step 2 results:", results
    #     self.terminate()
        
    # def terminate(self):
    #     print "controller.py: terminate"
    #     for solver in self.solvers.values():
    #         solver.callRemote("terminate").addErrback(self.failed, "Termination Failed")
    #     reactor.callLater(1, reactor.stop)
    #     return "Terminating remote solvers"

if __name__ == "__main__":
    controller = Controller()
    reactor.run()













    # Owner Number: 9448718185