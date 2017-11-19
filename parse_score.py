f = open('files/score.out')
date = f.readline()

i = 0
score = 0
for line in f.readlines():
    if i % 2 == 0:
        i += 1
        continue
    score += int(line.split(':')[1].split('/')[0])
    i += 1

print('score:', score)