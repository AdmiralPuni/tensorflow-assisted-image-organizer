import numpy

character_list = ['umi', 'umi', 'hanayo', 'eli', 'rin', 'nico', 'kotori']

new = numpy.unique(character_list)

print(' '.join(new))