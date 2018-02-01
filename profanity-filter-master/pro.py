'''import re
text='motherfucker fuck you ohhhh'
sum=0
c=0
for word in text.split():
    data=open("bad_words.txt","r")
    for line in data.readline():
        if word in line.split():
            x=line.split(",")
            #r=re.findall('\d+', line)
            sum=sum+x[1]
            c=c+1
            break



print(c)'''
text = ''
bad_word_rate_sum = 0
total_words = 0
bad_word_dict = {}


def init():
    f = open("bad_words.txt","r")
    for line in f:
        word, rating = line.split(",")
        rating = int(rating)
        bad_word_dict[word] = rating

def rating():
    f = open("transcript.txt")
    for line in f.readlines():
        words = line.split(' ')
        for word in words:
            global total_words
            total_words += 1
            if word in bad_word_dict:
                global bad_word_rate_sum
                bad_word_rate_sum += bad_word_dict[word]
    return bad_word_rate_sum / total_words


init()
print(rating())