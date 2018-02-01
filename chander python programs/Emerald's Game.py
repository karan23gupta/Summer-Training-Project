#https://www.hackerearth.com/challenge/college/efficiency/algorithm/emeralds-game/
x = int(input(''))
for i in range(x):
  abc = range(0,10)
  abc1 = []
  flag =1
  y = int(input(''))
  if y==0:
    print 'LOSE'
  else:
    j=1
    while flag:
      ab=y*j
      a = map(int,str(ab))
      leng =len(a)
      j=j+1
      for z in range(leng):
        abc1.append(a[z])
        if set(abc) == set(abc1):
          flag =0
          print ab
          break
        else:
          flag =1