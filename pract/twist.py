from twisted.internet import task
from twisted.internet import reactor
import datetime
import time






def main(args):
	for i in args:
		d = task.deferLater(reactor, i, f, "hello, world", i)
		d.addCallback(called)
		# print 'adding adding'

def f(s, d):
	curr_time= time.strptime(str(datetime.datetime.now().strftime("%H:%M:%S")), "%H:%M:%S")
	in_seconds = datetime.timedelta(hours=curr_time.tm_hour, minutes=curr_time.tm_min, seconds=curr_time.tm_sec).seconds
	print in_seconds
	return "Returning after %d seconds" % d

def called(result):
    print result

# curr_time= time.strptime(str(datetime.datetime.now().strftime("%H:%M:%S")), "%H:%M:%S")
# in_seconds = datetime.timedelta(hours=curr_time.tm_hour, minutes=curr_time.tm_min, seconds=curr_time.tm_sec).seconds
# print in_seconds
# d = task.deferLater(reactor, 3, f, "hello, world")
# d = task.deferLater(reactor, 5, f, "hello, world")



# f() will only be called if the event loop is started.
reactor.callWhenRunning(main, [10,3,5])
reactor.run()