# PLOTTING CURLICUE FRACTAL
# Please run with python 2.7.10

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as scp
import os
import sys
import subprocess
import shutil
import multiprocessing

def CurlieCueParallel(input_values):
	# Data-retrieval function that can be run in parallel
	s, n, job_index, number_of_indexes = input_values
	r = np.linspace(0,n-1,n)
	x = np.zeros(n)
	y = np.zeros(n)
	phi = np.zeros(n) # defined as the angle between the horizontal line an next line segment
	theta = np.zeros(n)

	iteration_array = np.arange(1,n+1)

	two_pi = 2*np.pi
	two_pi_s = two_pi*s

	for i,j in enumerate(iteration_array):

		theta[i] = (theta[i-1] + two_pi_s) % (two_pi)
		phi[i] = theta[i-1] + phi[i-1] % (two_pi)

		x[i] = x[i-1] + r[i-1]*np.cos(phi[i-1])
		y[i] = y[i-1] + r[i-1]*np.sin(phi[i-1])

	############ Progressbar for multiprocessing ############
	# if int(multiprocessing.current_process().name[-1]) == 1:
		# status = r'%3.2f%%' % job_index/float(number_of_indexes)*100
		# status += chr(8)*(len(status)+1)
		# print status,
		# sys.stdout.write('\r')
		# sys.stdout.write('%6.2f%%' % job_index/float(number_of_indexes))
		# sys.stdout.flush()

	return x, y, iteration_array

def PlotParallel(plot_input_values):
	# Function for plotting that can be run in parallel
	x_matrix, y_matrix, i, plot_config = plot_input_values
	new_xlim, new_ylim, folder_name, run_name = plot_config
	plt.cla()
	plt.plot(x_matrix,y_matrix,color="black")
	# Plot window settings
	plt.xlim(-new_xlim, new_xlim)
	plt.ylim(-new_ylim, new_ylim)

	# Saves figure in folder
	filename = '%s%s_%04d.png' % (folder_name, run_name, i)
	plt.savefig(filename)

class CurliecueFractal:
	def __init__(self,n,s):
		self.n = n
		self.s = s

	def _create_values(self, new_s = False, set_variables = True):
		n, s = self.n, self.s

		if new_s:
			s = new_s

		r = np.linspace(0,n-1,n)
		x = np.zeros(n)
		y = np.zeros(n)

		phi = np.zeros(n) # defined as the angle between the horizontal line an next line segment
		theta = np.zeros(n)

		iteration_array = np.arange(1,n+1)

		two_pi = 2*np.pi
		two_pi_s = two_pi*s

		for i,j in enumerate(iteration_array):

			theta[i] = (theta[i-1] + two_pi_s) % (two_pi)
			phi[i] = theta[i-1] + phi[i-1] % (two_pi)

			x[i] = x[i-1] + r[i-1]*np.cos(phi[i-1])
			y[i] = y[i-1] + r[i-1]*np.sin(phi[i-1])

		if set_variables:
			self.x, self.y, self.iteration_array = x, y, iteration_array

		return x, y, iteration_array

	def plot_fractal(self, filename_extension, s_value_string=False):
		print '====== Plotting fractal for %s' % filename_extension
		x, y, iteration_array = self._create_values()

		if not s_value_string:
			s_value_string = self.s

		self.s_value_string, self.filename_extension = s_value_string, filename_extension

		# plt.figure(figsize=(70,70)) # Is this usable?
		plt.plot(x,y,color='black')
		plt.title(r'Curliecue Fractal, $s=%s$, $n=%s$' % (s_value_string,self.n))
		plt.axis('equal')
		plt.savefig('curliecue_fractaly_s%s.eps' % filename_extension, format='eps', dpi=400)
		plt.close()

	def create_animation(self, gif_fps = False):
		"""Function for creating gifs. Removes all associated files afterwards."""
		x, y, iteration_array, filename_extension = (
			self.x, self.y, self.iteration_array, self.filename_extension)

		print '====== Creating animation fractal for %s' % filename_extension

		if not gif_fps: # setting a default gif time resolutions
			gif_time_resolution = int(self.n / 4.)
		else:
			gif_time_resolution = int(self.n / float(gif_fps))

		# Creating folder =======================
		folder_name = 'gif_%s_folder/' % filename_extension

		if os.path.isdir(folder_name): # Removing old files
			shutil.rmtree(folder_name)
		os.makedirs(folder_name) # Creating folder to store gif pictures in

		# Plotting and creating files ===========
		print 'Creating plots:'
		fig = plt.figure()
		ax = fig.add_subplot(111)
		fractal, = ax.plot(x,y,color='white') # Gets the line2d instance to update
		fractal.set_color('black')

		x_axis_limits = ax.get_xlim()
		y_axis_limits = ax.get_ylim()

		terminal_width = int(os.popen('stty size', 'r').read().split()[-1]) - 8 # Minus 8 to account for percentage symbol
		terminal_unit = terminal_width/float(iteration_array[-1])

		counter = 0

		for i,j in enumerate(iteration_array):
			if (i % gif_time_resolution) == 0:
				# Updates plot data
				fractal.set_xdata(x[0:i])
				fractal.set_ydata(y[0:i])
				
				# Draws new figure. Old figure is automatically removed?
				fig.canvas.draw()

				# Saves figure in folder
				fig.savefig('%s%s_%04.s.png' % (folder_name, filename_extension, counter), dpi=300)
		
				# Writes progressbar
				percentage_terminal = i*terminal_unit
				percentage_process = i/float(iteration_array[-1])*100
				sys.stdout.write('\r')
				sys.stdout.write('%6.2f%% %s' % (percentage_process, '#'*int(percentage_terminal)))
				sys.stdout.flush()

				# Old progress bar method
				# status = r'%3.2f%%' % (i/float(iteration_array[-1])*100)
				# status += chr(8)*(len(status)+1)
				# print status,

				counter += 1

		# Finalizes progress bar
		sys.stdout.write('\r100.00%% %s' % '#'*int(iteration_array[-1]*terminal_unit))

		# Updating the final plot ===============
		fractal.set_xdata(x[0:i])
		fractal.set_ydata(y[0:i])
		fig.canvas.draw()
		fig.savefig('%s%s_%04d.png' % (folder_name, filename_extension, i), dpi=300)

		print '\nPlot creation complete.'

		# Creating the gif ======================
		self._create_gif(folder_name, filename_extension)

	def _create_gif(self, folder_name, filename_extension):
		"""Function for creating a gif."""
		print 'Converting to gif...'
		subprocess.call('convert -delay 1 -loop 0 %s%s_*.png %s.gif' % (folder_name, filename_extension, filename_extension), shell=True)
		# print 'Deleting used folder with pictures: %s' % folder_name
		# shutil.rmtree(folder_name)
		print 'Gif creation done'

	def create_movie(self, folder_name, filename_extension):
		print 'Creating movie'
		
		subprocess.call('ffmpeg -framerate 24 -i %s%s_%%04d.png -c:v libx264 -r 24 %s.mp4' % (folder_name, filename_extension, filename_extension), shell=True)
		# If this fails, use:
		# ffmpeg -framerate 24 -pattern_type glob -i "*.png" -c:v libx264 -r 24 transformation_pi_1e-5.mp4
		# "ffmpeg -r 30 -i %s%s_%%04d.png -codec:v mpeg4 -flags:v +qscale -global_quality:v 0 -codec:a libmp3lame -r 24 %s.mp4"

		
		# print 'Deleting used folder with pictures: %s' % folder_name
		# shutil.rmtree(folder_name)
		print 'Movie %s created' % filename_extension

	def retrieve_data(self, s_value):
		x, y, iteration_array = self._create_values(s_value, set_variables = False)
		self.x_matrix[i] = x
		self.y_matrix[i] = y

	def transform(self, number1, number2, total_frames, run_name, movie=False):
		print '====== Transforming fractal for %s' % run_name
		s_values = np.linspace(number1,number2,total_frames)
		x_matrix = np.zeros((total_frames, self.n))
		y_matrix = np.zeros((total_frames, self.n))
		
		# Creating folder =======================
		folder_name = 'gif_%s_folder/' % run_name

		if os.path.isdir(folder_name): # Removing old files
			shutil.rmtree(folder_name)
		os.makedirs(folder_name) # Creating folder to store gif pictures in

		# Retrieving data =======================
		print 'Retrieving data:'

		# Parallelized data retrieval
		input_values = zip(s_values, [self.n for i in xrange(total_frames)], range(total_frames), [total_frames for i in xrange(total_frames)])
		pool = multiprocessing.Pool(processes=4)
		results = pool.map(CurlieCueParallel,input_values)

		for i in xrange(len(results)):
			x_matrix[i], y_matrix[i] = results[i][:2]

		# # Unparallelized method
		# for i in xrange(len(s_values)):
		# 	x, y, iteration_array = self._create_values(s_values[i], set_variables = False)
		# 	x_matrix[i] = x
		# 	y_matrix[i] = y

		# 	status = r'%3.2f%%' % (i/float(len(s_values))*100)
		# 	status += chr(8)*(len(status)+1)
		# 	print status,

		print 'Data retrieval complete.'

		# Old, non-parallelized method
		# # Plotting and creating files ===========
		# print 'Creating plots:'
		# fig = plt.figure()
		# ax = fig.add_subplot(111)

		# fractal, = ax.plot(x_matrix[-1],y_matrix[-1],color='white') # Gets the line2d instance to update

		# # # Plot window settings
		# plot_window_tolerance = 1.2 # Changes how much extra space we will view the window
		# new_xlim = np.max(np.abs(ax.get_xlim())) * plot_window_tolerance
		# new_ylim = np.max(np.abs(ax.get_ylim())) * plot_window_tolerance

		# ax.set_xlim(-new_xlim, new_xlim)
		# ax.set_ylim(-new_ylim, new_ylim)
		# fractal.set_color('black')

		# for i in xrange(total_frames):
		# 	#Updates plot data
		# 	fractal.set_xdata(x_matrix[i])
		# 	fractal.set_ydata(y_matrix[i])
			
		# 	# Draws new figure. Old figure is automatically removed?
		# 	fig.canvas.draw()

		# 	# Saves figure in folder
		# 	filename = '%s%s_%04d.png' % (folder_name, run_name, i)
		# 	fig.savefig(filename)

		# 	status = r'%3.2f%%' % (i/float(total_frames)*100)
		# 	status += chr(8)*(len(status)+1)
		# 	print status,

		# print 'Plot creation complete.'

		# Plotting and creating files - PARALLIZED ======================
		print 'Creating plots:'
		fig = plt.figure()
		ax = fig.add_subplot(111)

		fractal, = ax.plot(x_matrix[-1],y_matrix[-1],color='white') # Gets the line2d instance to update

		# Plot window settings
		plot_window_tolerance = 1.2 # Changes how much extra space we will view the window
		new_xlim = np.max(np.abs(ax.get_xlim()))*plot_window_tolerance
		new_ylim = np.max(np.abs(ax.get_ylim()))*plot_window_tolerance

		plot_config = [new_xlim, new_ylim, folder_name, run_name]
		plot_input_values = zip(x_matrix, y_matrix, range(total_frames), [plot_config for i in xrange(total_frames)])

		pool = multiprocessing.Pool(processes=4)
		results = pool.map(PlotParallel, plot_input_values)

		print 'Plot creation complete.'

		if movie:
			self.create_movie(folder_name, run_name)
		else:
			self._create_gif(folder_name, run_name)


N = 1e6

# s_transformation = CurliecueFractal(N,1)
# s_transformation.transform(np.pi-1e-5, np.pi+1e-5, 10000, 'transformation_pi_1e-5',movie=True)

# s_transformation2 = CurliecueFractal(N,1)
# s_transformation2.transform(scp.golden-1e-9,scp.golden,1000,'transformation3',movie=True)

# s_transformation2 = CurliecueFractal(N,1)
# s_transformation2.transform(np.pi - 1e-4, np.pi + 1e-4, 10000, 'transformation1', movie=True)


# s_transformation3 = CurliecueFractal(N,1)
# s_transformation3.transform(np.exp(1) - 1e-4, np.exp(1) + 1e-4, 10000, 'transformation_euler', movie=True)


# s_pi = CurliecueFractal(N,np.pi)
# s_pi.plot_fractal('pi_tests','\pi_test')
# s_pi.create_animation(gif_fps=20)

# s_golden_ratio = CurliecueFractal(N,scp.golden)
# s_golden_ratio.plot_fractal('golden_ratio','\phi')
# s_golden_ratio.create_animation(gif_fps=60)

# s_ln2 = CurliecueFractal(N,np.log(2.0))
# s_ln2.plot_fractal('natural_log2','\ln 2')
# s_ln2.create_animation(gif_fps=60)

s_euler = CurliecueFractal(N,np.exp(1))
s_euler.plot_fractal('eulers_number','e')
# s_euler.create_animation(gif_fps=60)

# s_sqrt2 = CurliecueFractal(N,np.sqrt(2))
# s_sqrt2.plot_fractal('sqrt2','\sqrt{2}')
# s_sqrt2.create_animation(gif_fps=60)

# Euler_Mascheroni = 0.5772156649015328606
# s_Euler_Mascheroni = CurliecueFractal(N, Euler_Mascheroni)
# s_Euler_Mascheroni.plot_fractal('Euler_Mascheroni','\gamma')
# s_Euler_Mascheroni.create_animation(gif_fps=60)

# s_sqrt3 = CurliecueFractal(N,np.sqrt(3))
# s_sqrt3.plot_fractal('sqrt3','\sqrt{3}')
# s_sqrt3.create_animation(gif_fps=60)

# reduced_planck = 6.582119514
# s_reduced_planck = CurliecueFractal(N,reduced_planck)
# s_reduced_planck.plot_fractal('hbar','\hbar')
# s_reduced_planck.create_animation(gif_fps=60)

# catalans_constant = 0.915965594177219015054603514932384110774
# s_catalans_constant = CurliecueFractal(N,catalans_constant)
# s_catalans_constant.plot_fractal('catalans_constant','G=\sum^\infty_{n=0} \\frac{(-1)^n}{(2n+1)^2}')
# s_catalans_constant.create_animation(gif_fps=60)