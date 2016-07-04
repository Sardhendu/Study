from twist_processes import WCProcessProtocol
from twisted.internet import reactor


wcProcess = WCProcessProtocol("accessing protocols through Twisted is fun!\n")
reactor.spawnProcess(wcProcess, 'wc', ['wc'])
reactor.run()