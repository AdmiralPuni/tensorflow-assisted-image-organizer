import os

charlist = []

for dir in os.listdir('images/hololive/train'):
    charlist.append(dir.replace('-',' ').capitalize())

print(', '.join(charlist))