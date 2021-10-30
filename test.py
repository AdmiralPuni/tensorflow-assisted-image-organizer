import os

charlist = []

for dir in os.listdir('images/nijigasaki/'):
    charlist.append(dir.replace('-',' ').capitalize())

print('", "'.join(charlist))