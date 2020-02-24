import tensorflow as tf
import numpy as np
from sympy import *
from sympy.geometry import intersection
from sklearn.preprocessing import MinMaxScaler

class PropertyConfigurationGeneratorModel():

  @staticmethod
  def resolve_generating_functions(self, generating_functions):
    return tf.real(generating_functions), tf.imag(generating_functions)
  
  # series = indexes based on A-A, B-B generating functions, outputs series for the factor model
  @staticmethod
  def summation_series(series, generating_functions):
    i = 0
    for s in series:
      yield tf.reduce_sum(tf.convert_to_tensor(generating_functions[:, i:i+s], dtype=tf.complex64), axis=1)
      i += s
  
  # means based on A-A, B-B computing from the result
  @staticmethod
  def summation_posterior(series, generator_model):
    i = 0
    for s in series:
      yield tf.reduce_sum(tf.convert_to_tensor(generator_model.model.means_[:, i:i+s], dtype=tf.float32), axis=1)
      i += 1

class ProximityConfigurationGeneratorModel():

  coordinate_space = ['x', 'y']
  # defining cardinal directions as ['N', 'E', 'W', 'S']
  cardinal_space = [Segment2D(Point2D(0,0), Point2D(0,1)), Segment2D(Point2D(0,0), Point2D(1,0)), Segment2D(Point2D(0,0), Point2D(0,-1)), \
  Segment2D(Point2D(0,0), Point2D(0,-1))]

  def __init__(self):
    pass

  # space = coordinate space
  def evaluate_generating_functions(self, polygon, spaces, elements, centroids, distance):
    
    class Centroid():

      def __init__(self, centroid, dst, axis, polygon):
        self.centroid = centroid
        self.dst = dst
        self.axis = axis
        self.polygon = polygon

      def run(self):
        a = map(self.fn_map, dst)
        seg = map(self.fn_segment, a)
        self.A_g_t = tf.map_fn(self.fn_cos_gen, seg)
        self.B_g_t = tf.map_fn(self.fn_sin_gen, seg)

      def fn_point(self, l):
        return Segment2D(self.centroid, l)
      
      def fn_map(self, d):
        c = Circle(self.centroid, d)
        l_int = intersection(self.polygon, c)
        if(len(l_int) > 0):
          return map(self.fn_point, l_int) # returns list of segments
        else:
          return None

      def fn_segment(self, elem):
        all_g_t = []
        if elem is None:
          return all_g_t
        if(len(elem) > 1):
          g_t = []
          for j in range(1, len(elem)):
            seg = Segment2D(Point2D(elem[j-1]), Point2D(elem[j]))
            if(not self.polygon.contains(seg)): # the segment is not part of the polygon sides
              l1 = Segment2D(self.centroid, elem[j-1])
              l2 = Segment2D(self.centroid, elem[j])
              angle1 = self.axis.angle_between(l1)
              angle2 = self.axis.angle_between(l2)
              g_t.append(tuple(angle1, angle2))
          all_g_t.append(g_t)
        return all_g_t

      # {An}
      def fn_sin_calc(self, elem):
        angle1 = elem[0]
        angle2 = elem[1]
        k = 2*np.pi*idx/(angle2 - angle1)
        return constant / idx * (angle2 - angle1) * (sin(k*angle2) - sin(k*angle1))
      
      # {Bn}
      def fn_cos_calc(self, elem):
        angle1 = elem[0]
        angle2 = elem[1]
        k = 2*np.pi*idx/(angle2 - angle1)
        return constant / idx * (angle2 - angle1) * (cos(k*angle1) - cos(k*angle2))

      def fn_sin_gen(self, idx, b_g_t):
        if(len(b_g_t) > 0):
          return tf.reduce_sum(tf.map_fn(self.fn_cos_calc, b_g_t), axis=1)
        else:
          return 0

      def fn_cos_gen(self, idx, a_g_t):
        if(len(a_g_t) > 0):
          return tf.reduce_sum(tf.map_fn(self.fn_sin_calc, a_g_t), axis=1)
        else:
          return 0
    
    axes = [Line2D(centroid, Point2D(centroid.x + 100.0, centroid.y)) for centroid in centroids]
    dst = tf.lin_space(0, distance, elements)
    dst = [1 / tf.square(d) for d in dst]

    cent = [Centroid(centroid, d, axis, polygon) for centroid, d, axis in zip(centroids, dst, axes)]

    c.run() for c in cent

    return [c.g_t for c in cent]