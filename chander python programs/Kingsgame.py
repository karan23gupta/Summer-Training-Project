#https://www.hackerearth.com/practice/algorithms/sorting/quick-sort/practice-problems/algorithm/kings-race-8/activity/
t = int(raw_input())
for i in range(t):
    n, k = map(int, raw_input().split())
    js = map(int, raw_input().split())
    hs = map(int, raw_input().split())
    
    c = 0
    t = 0
    
    maxl1=max(js)
    maxl2=max(hs)
    if maxl2>maxl1:
		for u in hs:
			if u > maxl1:
				maxl2=u
				break
		hs=hs[:hs.index(maxl2)]
		if len(hs)==0:
			maxl2=0
		else:
			maxl2=max(hs)
    count=0
    
    while js[count]<maxl2:
		count+=1
    print count