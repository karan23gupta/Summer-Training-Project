#!/usr/bin/env python
import sys


bad_words = []
threshold = 5
def init():
	file = open("assets/data/bad_words.txt",'r')
	for line in file:
		line = line.strip()
		word = (line.split(' '))[0]
		bad_words.append(word)
def main():
	bad_count= 0
	init()
	for line in sys.stdin:
		line = line.strip()
		line = line.split()
		word , count = line[0],line[1]
		count = int(count)
		if word in bad_words:
			bad_count+=count
	if bad_count> threshold:
		print 'User is flagged with %d badwords' % (bad_count)
	else:
		print 'User is clean'
    
if __name__ == "__main__":
  main()

