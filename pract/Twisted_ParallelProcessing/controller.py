"""
Controller.py
Original version by Bruce Eckel
Starts and manages solvers in separate processes for parallel processing.
"""
import sys
from subprocess import Popen
from twisted.spread import pb
from twisted.internet import reactor, defer

START_PORT = 8800#8686#5566
MAX_PROCESSES = 2

class Controller(object):

    def broadcastCommand(self, remoteMethodName, arguments, nextStep, failureMessage):
        print "controller.py: broadcasting..."
        # print 'solver solver solver ', self.solver#.values()
        deferreds = [solver.callRemote(remoteMethodName, arguments) 
                     for solver in self.solvers.values()]
        print "controller.py: broadcasted"
        reactor.callLater(3, self.checkStatus)

        defer.DeferredList(deferreds, consumeErrors=True).addCallbacks(
            nextStep, self.failed, errbackArgs=(failureMessage))
    
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
        self.solvers = dict.fromkeys([("localhost", i) for i in range(START_PORT, START_PORT+MAX_PROCESSES)])
        print 'The solvers are: ', self.solvers 
        # self.pids = [Popen(["python", "solver.py", str(port)]).pid for ip, port in self.solvers]   # This command will run the solver file
        self.pids = [] 
        for ip, port in self.solvers:
            a = Popen(["python", "solver.py", str(port)]).pid
            self.pids.append(a)
        print "PIDS: ", self.pids
        self.connected = False
        reactor.callLater(5, self.connect)

    def connect(self):
        print "controller.py: connect"
        connections = []
        for address, port in self.solvers:
            print 'THe address and port are: ', address, port
            factory = pb.PBClientFactory()
            reactor.connectTCP(address, port, factory)
            connections.append(factory.getRootObject())
        defer.DeferredList(connections, consumeErrors=True).addCallbacks(
            self.storeConnections, self.failed, errbackArgs=("Failed to Connect"))

        print "controller.py: starting parallel jobs"
        self.start()

    def storeConnections(self, results):
        print "controller.py: storeconnections"
        for (success, solver), (address, port) in zip(results, self.solvers):
            self.solvers[address, port] = solver
        print "controller.py: Connected; self.solvers:", self.solvers
        self.connected = True

    def start(self):
        print "controller.py: Begin the solving process"
        if not self.connected:
            print 'Not connected, trying again'
            return reactor.callLater(0.5, self.start)
        print 'Connected, Moving ahead'
        self.broadcastCommand("step1", ("step 1"), self.step2, "Failed Step 1")

    def step2(self, results):
        print "controller.py: step 1 results:", results
        self.broadcastCommand("step2", ("step 2"), self.step3, "Failed Step 2")

    def step3(self, results):
        print "controller.py: step 2 results:", results
        self.broadcastCommand("step3", ("step 3"), self.collectResults, "Failed Step 3")

    def collectResults(self, results):
        print "controller.py: step 3 results:", results
        self.terminate()
        
    def terminate(self):
        print "controller.py: terminate"
        for solver in self.solvers.values():
            solver.callRemote("terminate").addErrback(self.failed, "Termination Failed")
        reactor.callLater(1, reactor.stop)
        return "Terminating remote solvers"

if __name__ == "__main__":
    controller = Controller()
    reactor.run()