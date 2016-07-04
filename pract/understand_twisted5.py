from twisted.internet import defer, reactor
from twisted.web.client import getPage

urls = [
  'http://yahoo.com',
  'http://www.google.com',
  'http://www.google.com/MicrosoftRules.html',
  'http://bogusdomain.com',
  ]

def pageCallback(result, url):
  data = {
    'length': len(result),
    'content': result[:10],
    'url': url,
    }
  return data

def pageErrback(error, url):
  return {
    'msg': error.getErrorMessage(),
    'err': error,
    'url': url,
    }

def getPageData(url):
  d = getPage(url, timeout=5)
  d.addCallback(pageCallback, url)
  d.addErrback(pageErrback, url)
  return d

def listCallback(result):
  for ignore, data in result:
    if data.has_key('err'):
      print "Call to %s failed with data %s" % (data['url'], str(data))
    else:
      print "Call to %s succeeded with data %s" % (data['url'], str(data))

def finish(ign):
  reactor.stop()

def test():
  deferreds = []
  for url in urls:
    d = getPageData(url)
    deferreds.append(d)
  dl = defer.DeferredList(deferreds, consumeErrors=1)
  dl.addCallback(listCallback)
  dl.addCallback(finish)

test()
reactor.run()