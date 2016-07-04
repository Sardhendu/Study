

####################################  staticmethod   #######################################
class Date:
  def __init__(self, month, day, year):
  	print 'Inside Init'
  	self.month = month
  	self.day   = day
  	self.year  = year


  def display(self):
  	print 'Inside display'
  	return "{0}-{1}-{2}".format(self.month, self.day, self.year)


  @staticmethod
  def millenium(month, day):				
  	print 'Inside millenium'
  	return Date(month, day, 2000)

new_year = Date(1, 1, 2013)               	# Creates a new Date object
print ''
millenium_new_year = Date.millenium(1, 1) 	# also creates a Date object.  goes first to the millenium method, then init to create a factory
print ''
print 'Objects created'
print ''
# Proof:
print new_year.display()           			# "1-1-2013"
print millenium_new_year.display() 			# "1-1-2000"	# Accesses the default value of year (2000)

print 'new_year: Am I an Object of Date: ', isinstance(new_year, Date) 			# True
print 'millenium_new_year: Am I an Object of Date: ', isinstance(millenium_new_year, Date) 	# True


print ''
print ''

class DateTime(Date):
	def display(self):
		return "{0}-{1}-{2} - 00:00:00PM".format(self.month, self.day, self.year)


datetime1 = DateTime(10, 10, 1990)			# Goes to the init of Data (since DataTime inherits it and displays year as 1990)
datetime2 = DateTime.millenium(10, 10)		# Goes to the millenium od Data 

print 'datetime1: Am I an Object of DateTime: ', isinstance(datetime1, DateTime) # True
print 'datetime2: Am I an Object of DateTime: ',isinstance(datetime2, DateTime) # False
print ''
print datetime1.display() # returns "10-10-1990 - 00:00:00PM"
print datetime2.display() # returns "10-10-2000" because it's not a DateTime object but a Date object. Check the implementation of the millenium method on the Date class


print ''
print ''
print ''
print ''


####################################  classmethod   #######################################
class Date:
  def __init__(self, month, day, year):
  	print 'Inside Init'
  	self.month = month
  	self.day   = day
  	self.year  = year


  def display(self):
  	print 'Inside display'
  	return "{0}-{1}-{2}".format(self.month, self.day, self.year)


  @classmethod
  def millenium(cls, month, day):				
  	print 'Inside millenium'
  	return cls(month, day, 2000)

new_year = Date(1, 1, 2013)               	# Creates a new Date object
print ''
millenium_new_year = Date.millenium(1, 1) 	# also creates a Date object.  goes first to the millenium method, then init to create a factory
print ''
print 'Objects created'
print ''
# Proof:
print new_year.display()           			# "1-1-2013"
print millenium_new_year.display() 			# "1-1-2000"	# Accesses the default value of year (2000)

print 'new_year: Am I an Object of Date: ', isinstance(new_year, Date) 			# True
print 'millenium_new_year: Am I an Object of Date: ', isinstance(millenium_new_year, Date) 	# True


print ''
print ''

class DateTime(Date):
	def display(self):
		return "{0}-{1}-{2} - 00:00:00PM".format(self.month, self.day, self.year)


datetime1 = DateTime(10, 10, 1990)			# Goes to the init of Data (since DataTime inherits it and displays year as 1990)
datetime2 = DateTime.millenium(10, 10)		# Goes to the millenium od Data 

print 'datetime1: Am I an Object of DateTime: ', isinstance(datetime1, DateTime) # True
print 'datetime2: Am I an Object of DateTime: ',isinstance(datetime2, DateTime) # False
print ''
print datetime1.display() # returns "10-10-1990 - 00:00:00PM"
print datetime2.display() # returns "10-10-2000" because it's not a DateTime object but a Date object. Check the implementation of the millenium method on the Date class