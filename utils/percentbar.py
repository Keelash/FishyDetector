import sys

class PercentBar:
	def __init__(self, title = "Current Operation"):
		self.currPercent = 0
		self.title = title

	def setPercent(self, percent) :
		self.currPercent = percent

	def show(self) :
		sys.stdout.write('\r')
		sys.stdout.write(self.title + " : [%-20s] %d%%" % ('='*(self.currPercent/5), self.currPercent))
		sys.stdout.flush()