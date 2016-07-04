from twisted.internet import defer, reactor
from twisted.web.client import getPage

def pageCallback(result):
  return len(result)

def listCallback(result):
  print result

def finish(ign):
  reactor.stop()

def test():
  d1 = getPage('http://www.google.com')
  d1.addCallback(pageCallback)
  d2 = getPage('http://yahoo.com')
  d2.addCallback(pageCallback)
  dl = defer.DeferredList([d1, d2])
  dl.addCallback(listCallback)
  dl.addCallback(finish)

test()
reactor.run()