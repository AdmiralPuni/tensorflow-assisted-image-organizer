import os

charlist = []

for dir in os.listdir('images/hololive-all/train'):
    charlist.append(dir.replace('-',' ').capitalize())

print('", "'.join(charlist))