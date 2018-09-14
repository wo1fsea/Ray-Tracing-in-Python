# -*- coding: utf-8 -*-
"""----------------------------------------------------------------------------
Author:
    Huang Quanyong (wo1fSea)
    quanyongh@foxmail.com
Date:
    2018/9/13
Description:
    ray_tracing_in_python.py
----------------------------------------------------------------------------"""

import threading
from PIL import Image
import numpy as np
import numpy.random as random
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import QTimer

screen_size = (256, 512)
height, width = screen_size


def render_thread(render_buff, width, height):
	class Camera(object):
		def __init__(self):
			self.origin = np.array((0, 0, 0))
			self.horizontal = np.array((4, 0, 0))
			self.vertical = np.array((0, 2, 0))

			self.screen_origin = np.array((-2, -1, -1))

		def get_ray(self, u, v):
			return Ray(self.origin, self.screen_origin + self.horizontal * u + self.vertical * v)

	class Ray(object):
		def __init__(self, src, dst):
			delta = (dst - src)
			self.k = delta / np.linalg.norm(delta)
			self.b = src

		@property
		def origin(self):
			return self.b

		@property
		def direction(self):
			return self.k

		def point_at_t(self, t):
			return self.b + t * self.k

	class HitPoint(object):
		def __init__(self, ray, obj, t, point, normal):
			self.ray = ray
			self.obj = obj
			self.t = t
			self.point = point
			self.normal = normal

	class Sphere(object):
		def __init__(self, center, radius):
			self.center = center
			self.radius = radius

		def find_hit_point(self, ray):
			oc = ray.origin - self.center
			a = np.dot(ray.direction, ray.direction)
			b = np.dot(oc, ray.direction)
			c = np.dot(oc, oc) - np.dot(self.radius, self.radius)
			discriminant = b * b - a * c

			hit_points = []

			if discriminant > 0:
				t1 = (- b - np.sqrt(discriminant)) / a
				p1 = ray.point_at_t(t1)
				n1 = (p1 - self.center) / self.radius
				t2 = (- b + np.sqrt(discriminant)) / a
				p2 = ray.point_at_t(t2)
				n2 = (p2 - self.center) / self.radius
				hit_points.append(HitPoint(ray, self, t1, p1, n1))
				hit_points.append(HitPoint(ray, self, t2, p2, n2))

			return hit_points

	def find_hit_point(obj_list, ray):
		hit_points = []
		for obj in obj_list:
			hits = obj.find_hit_point(ray)
			hit_points.extend(hits)

		return hit_points

	def random_in_unit_sphere():
		while True:
			r = (2 * random.random(3) - np.array((1, 1, 1)))
			if np.dot(r, r) <= 1:
				return r

	def color(obj_list, ray, depth):
		if depth < 0:
			return np.array((1, 1, 1))

		hit_points = find_hit_point(obj_list, ray)
		hit_points = list(filter(lambda x: x.t > 0.001, hit_points))

		if hit_points:
			hit_points.sort(key=lambda x: x.t)
			hit_point = hit_points[0]

			target = hit_point.point + hit_point.normal + random_in_unit_sphere()
			return 0.5 * color(obj_list, Ray(hit_point.point, target), depth - 1)

		t = 0.5 * (ray.direction[1] + 1.)
		c = (1. - t) * np.array((1., 1., 1.)) + t * np.array((0.5, 0.7, 1))
		return c

	depth = 2
	samples = 10

	obj_list = [Sphere(np.array((0, -100.5, -1.)), 100), Sphere(np.array((0, 0, -1.)), 0.5)]

	camera = Camera()

	for x in range(width):
		for y in range(height):
			c = 0
			for s in range(samples):
				u = (x + random.random()) / width
				v = (y + random.random()) / height
				ray = camera.get_ray(u, v)
				c += color(obj_list, ray, depth)

			c = c / samples
			c = np.sqrt(c)
			c = (c * 0xff).astype(np.uint8)
			render_buff[height - 1 - y][x] = 0xff000000 + 0x10000 * c[2] + 0x100 * c[1] + c[0]


render_buff = np.ones(screen_size, dtype=np.uint32)
t = threading.Thread(target=render_thread, args=(render_buff, width, height))
t.start()


class App(QMainWindow):
	def __init__(self):
		super(App, self).__init__()
		self.title = 'monitor'
		self.left = 0
		self.top = 0
		self.width = 640
		self.height = 480
		self.init()

		self._last_time = 0
		self._intervals = []

	def init(self):
		self.setWindowTitle(self.title)

		# Create widget
		self.label = QLabel(self)
		self.timer = QTimer(self)
		self.timer.timeout.connect(self.timeOut)
		self.timer.start(10)
		self.show()

	def timeOut(self):
		frame = Image.fromarray(render_buff, mode="RGBA")
		pixmap = frame.toqpixmap()
		self.label.setPixmap(pixmap)
		self.label.resize(pixmap.width(), pixmap.height())
		self.resize(pixmap.width(), pixmap.height())


t = threading.Thread(target=render_thread, args=(render_buff, width, height))
t.start()
app = QApplication(sys.argv)
ex = App()
sys.exit(app.exec_())
