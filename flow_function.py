import tensorflow as tf
import numpy as np
from sympy import *
from sympy.geometry import intersection

class CirculationConfigurationGenerationEntity():

  D = 0.219 * pow(10, -4)
  delta_sq_ua = 1.0

  def __init__(self):
    self.x, self.y = Symbols('x y')
    pass

  def evaluate_flow_function(self, pmax, pmin, dmax):
    u, v = Symbols('u v')
    u = 2*sqrt(2)*D / dmax**2 * (pmax*dmax**2 - (pmax - pmin)*(self.x**2 + self.y**2))**(5/2) * (self.x**2 - self.y**2) / (self.x**2 + self.y**2)
    v = 2*sqrt(2)*D / dmax**2 * (pmax*dmax**2 - (pmax - pmin)*(self.x**2 + self.y**2))**(5/2) * 2*self.x*self.y / (self.x**2 + self.y**2)
    return u, v
  
  # space = coordinate space
  def evaluate_parameters(self, pmin, pmax, dmax):
    u, v = self.evaluate_flow_function(pmin, pmax, dmax)

    sigma_x = integrate(u, (self.x))
    sigma_y = integrate(v, (self.y))
    tau_x = integrate(v, (self.x))
    tau_y = integrate(u, (self.y))

    return sigma_x, sigma_y, tau_x, tau_y

  def evaluate_generating_functions(self, polygon, spaces, elements, centroids, distance):
    sigma_x, sigma_y, tau_x, tau_y = self.evaluate_parameters()
    sigma = sigma_x + sigma_y
    tau = tau_y - tau_x

    class Centroid():

      def __init__(self, centroid, dst, axis, polygon, qty):
        self.centroid = centroid
        self.dst = dst
        self.axis = axis
        self.polygon = polygon
        self.qty = qty

      def run(self):
        self.a = map(self.fn_map, dst)
        seg = map(self.fn_segment, a)
        self.A_g_t = map(self.fn_cos_gen, seg)
        self.B_g_t = map(self.fn_sin_gen, seg)

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
      def fn_sin_calc(self, idx, elem):
        angle1 = elem[0]
        angle2 = elem[1]
        k = 2*np.pi*idx/(angle2 - angle1)
        C = self.dst[idx] * Curve((sin(t)*sin(kt), cos(t)*sin(kt)), (t, angle1, angle2))
        v = line_integrate(self.qty, C, [x, y])
        return v
      
      # {Bn}
      def fn_cos_calc(self, idx, elem):
        angle1 = elem[0]
        angle2 = elem[1]
        k = 2*np.pi*idx/(angle2 - angle1)
        C = self.dst[idx] * Curve((sin(t)*cos(kt), cos(t)*cos(kt)), (t, angle1, angle2))
        v = line_integrate(self.qty, C, [x, y])
        return v

      def fn_sin_gen(self, idx, b_g_t):
        if(len(b_g_t) > 0):
          return tf.reduce_sum(tf.map_fn(self.fn_cos_calc, (np.ones(len(b_g_t))*idx, b_g_t)), axis=1)
        else:
          return 0

      def fn_cos_gen(self, idx, a_g_t):
        if(len(a_g_t) > 0):
          return tf.reduce_sum(tf.map_fn(self.fn_sin_calc, (np.ones(len(a_g_t))*idx, a_g_t)), axis=1)
        else:
          return 0

    qty = [sigma, tau]
    axes = [Line2D(centroid, Point2D(centroid.x + 100.0, centroid.y)) for centroid in centroids]
    dst = tf.lin_space(0, distance, elements)
    dst = [1 / tf.square(d) for d in dst]

    cent = [Centroid(centroid, d, axis, polygon, q) for centroid, d, axis in zip(centroids, dst, axes) for q in qty]

    c.run() for c in cent

    pass