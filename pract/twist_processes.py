from twisted.internet import protocol

class WCProcessProtocol(protocol.ProcessProtocol):

    def __init__(self, text):
        self.text = text


    def connectionMade(self):
        self.transport.write(self.text)
        self.transport.closeStdin()


    def outReceived(self, data):
        fieldLength = len(data) / 3
        lines = int(data[:fieldLength])
        words = int(data[fieldLength:fieldLength*2])
        chars = int(data[fieldLength*2:])
        self.transport.loseConnection()
        self.receiveCounts(lines, words, chars)

    def receiveCounts(self, lines, words, chars):
        print 'Received counts from wc.'
        print 'Lines:', lines
        print 'Words:', words
        print 'Characters:', chars

# class WCProcessProtocol(protocol.ProcessProtocol):

#     def __init__(self, text):
#         self.text = text


#     def connectionMade(self):
#         self.transport.write(self.text)
#         self.transport.closeStdin()


#     def outReceived(self, data):
#         fieldLength = len(data) / 3
#         lines = int(data[:fieldLength])
#         words = int(data[fieldLength:fieldLength*2])
#         chars = int(data[fieldLength*2:])
#         self.transport.loseConnection()
#         self.receiveCounts(lines, words, chars)

#     def receiveCounts(self, lines, words, chars):
#         print 'Received counts from wc.'
#         print 'Lines:', lines
#         print 'Words:', words
#         print 'Characters:', chars