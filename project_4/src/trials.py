
import PIL
import tensorflow as tf
import numpy

from scipy.spatial.distance import pdist

a = numpy.array([1,2,3,4,5])
b = numpy.array([12,32,43,45,52])
c = numpy.array([143,31,63,42,92])

print(pdist(numpy.array([a,b])))
print(pdist(numpy.array([a,c])))