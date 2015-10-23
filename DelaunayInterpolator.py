# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 15:37:05 2015

@author: gkocher

WORKING FINE FOR THIS SETUP:
Python 2.7.6 |Anaconda 1.9.1 (64-bit)| (default, Nov 11 2013, 10:49:15) [MSC v.1500 64 bit (AMD64)] on win32
Imported NumPy 1.9.3, SciPy 0.16.0, Matplotlib 1.4.3
"""



#==============================================================================
# Defines the DelaunayInterpolator class and tests on simple demo.
# Technically the interpolation function could be done on data with any dimensionality,
# not just 3D. However, the plotting code requires 3D. Also, a few internal 
# dimensionality checks are done that assume you are working in 3D space.

#To use with your data (only 5 or 6 lines):
#from DelaunayInterpolator import DelaunayInterpolator
#data = np.array([x_sampled,y_sampled,z_sampled,]).T #array with 3 column vectors of x,y,z points
#DelaunaySurface = DelaunayInterpolator(data)
#xy_pairs = np.array([xpoints,ypoints]).T #array with 2 column vectors of x,y points where you want to interpolate
#interpolated_z = DelaunaySurface.interpolate(xy_pairs)
#DelaunaySurface.plot3D()
#==============================================================================

import os
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D





class DelaunayInterpolator():
    
    """
    Take in an array of sample points, xyz_stack.
    
    interpolate:
    Return the Delaunay interpolated z coordinate at every input (x,y) pair.
    
    plot3D:
    Plot a 3D surface. If there are a few hundred thousand or few million points,
    the resulting figure can be extremeley slow to respond to user input like rotations.
    So instead, can just save a few figures at predefined viewing orientations.
    """
    
    
    
    def __init__(self,xyz_stack):
        #xyz_stack should be Npoints x 3 NumPy array
    
        #Check xyz_stack is a NumPy array with correct dimensions and shape:
        if type(xyz_stack) != type(np.zeros(0)) or (xyz_stack.ndim != 2)\
        or (xyz_stack.shape[1] != 3) or (xyz_stack[:,0].size != xyz_stack[:,1].size != xyz_stack[:,2].size):
            message = 'xyz_stack needs to be a 2 dimensional NumPy array. It is Npoints x 3.'\
            + 'xyz_stack[:,0] is the array of x coordinates, xyz_stack[:,1] is the '\
            + 'array of y coordinates, and xyz_stack[:,2] is the array of z coordinates.'
            raise ValueError(message)
            
        self.xyz_stack = xyz_stack
        self.Npoints = self.xyz_stack.shape[0]
        self.x_array = self.xyz_stack[:,0]
        self.y_array = self.xyz_stack[:,1]
        self.z_array = self.xyz_stack[:,2]
        self.tri = Delaunay(np.array([self.x_array,self.y_array]).T)#possibly want delaunay kwargs..., e.g. 'Qt' #for options see: http://www.qhull.org/html/qdelaun.htm
        self.interpolated_Z_array = 'None'#Why this instead of None without quotes? Because later line doing: if self.interpolated_Z_array != ... gives warning about future version
        self.save_dir = None
        self.base_filename = None
        self.XY_cols = None

        
        
        
    def interpolate(self,XY_cols):
        """
        Return the interpolated z coordinates for all given (x,y) pairs.
        
        XY_cols: an Npoints x 2 NumPy array of the points to interpolate.
        First column is array of x coordinates of the points to interpolate,
        second column is array of y coordinates of the points to interpolate. 
        Both columns have Npoints elements.
        """

        self.XY_cols = XY_cols
        
        #Code in this function below follows from:
        # http://stackoverflow.com/questions/30373912/interpolation-with-delaunay-triangulation-n-dim?lq=1
        
        # dimension of the problem
        n = 2

        # find simplices that contain interpolated points
        s = self.tri.find_simplex(self.XY_cols)
        # get the vertices for each simplex
        verts = self.tri.vertices[s]
        # get transform matrices for each simplex (see explanation below)
        m = self.tri.transform[s]
        # for each interpolated point p (every row of self.XY_cols), mutliply the transform matrix by 
        # vector p-r, where r=m[:,n,:] is one of the simplex vertices to which 
        # the matrix m is related to (again, see bellow)
        b = np.einsum('ijk,ik->ij', m[:,:n,:n], self.XY_cols-m[:,n,:])
        
        # get the weights for the vertices; `b` contains an n-dimensional vector
        # with weights for all but the last vertices of the simplex
        # (note that for n-D grid, each simplex consists of n+1 vertices);
        # the remaining weight for the last vertex can be copmuted from
        # the condition that sum of weights must be equal to 1
        weights = np.c_[b, 1-b.sum(axis=1)]

        #Get the array of interpolated heights
        self.interpolated_Z_array = np.einsum('ij,ij->i', self.z_array[verts], weights)  
        
        #The linked post mentions tri.find_simplex(...) will return -1 for points 
        #outside of the convex hull of the triangulation. Since they are outside,
        #just give them NaN values to make it clear they are not successfully interpolated
        self.interpolated_Z_array[np.where(s==-1)[0]] = np.nan
        
        return self.interpolated_Z_array




    def plot3D(self,lifted_height=0.,close_plot=False,save_plot=False,save_dir='',base_filename=''):
        """
        Plot the figure at a few orientations and optionally save those figures
        
        """

        #Some Matplotlib 3D viewing parameters:
        figsize = (18,14)
        elev_array = np.array([50.])#This probably doesn't generalize well when graphs have different z values. So just don't do his and only vary the angles
        azim_array = np.linspace(0.,360.-30.,360./30.)#np.expand_dims(np.linspace(0.,360.-45.,360./45.),axis=1)
        el,az = np.meshgrid(elev_array,azim_array)
        orientation_array = np.vstack((el.flatten(),az.flatten()))

        #For every orientation, make a 3D surface plot of the Delaunay Triangulated surface.
        #Overlay 2 sets of points:
        #One set is the (x,y) points in a plane above the surface,
        #the other set is the points at the specified (x,y) coordinates projected onto the Delaunay surface.
        #This second set of points is the actual interpolated height.
        for kk in xrange(orientation_array.shape[1]):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            surf = ax.plot_trisurf(self.x_array, self.y_array, self.z_array, triangles=self.tri.simplices, cmap=plt.cm.Spectral,linewidth=.1,alpha=.5) #linewidth = .01, .1 depending on nsimplices
            plt.xlabel('X',fontsize=30)
            plt.ylabel('Y',fontsize=30)
            #plt.zlabel('Z',fontsize=30)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            
            #If an interpolation has been done:
            #Plot the x,y point at z=0, and then also projection onto surface at interpolated z,
            #and connect the point and its projection with a line:
            #If no interpolation has been done, will just plot the 3D surface without the points.
            if self.interpolated_Z_array != 'None':
                markersize = 20
                #Plot the x,y points above the surface:
                ax.scatter(self.XY_cols[:,0], self.XY_cols[:,1], lifted_height, s=markersize, c=u'r', depthshade=False)
                #Plot the points at their interpolated heights (should look like the above points projected onto the surface):
                ax.scatter(self.XY_cols[:,0], self.XY_cols[:,1], self.interpolated_Z_array, s=markersize, c=u'k', depthshade=False)
                #Connect those points with a line to make more visible:
                for point in xrange(self.interpolated_Z_array.size):
                    x = np.repeat(self.XY_cols[point,0],2)
                    y = np.repeat(self.XY_cols[point,1],2)
                    z = np.array([self.interpolated_Z_array[point],lifted_height])
                    ax.plot(x,y,z,color='k')
                
                
            #Flip the y axis so it is in same oritentation as a normal 2D Python Matplotlib imshow
            plt.gca().invert_yaxis()
            
            #Specify the viewing orientation:
            elev = orientation_array[0,kk]#This probably doesn't generalize well when graphs have different z values. So just don't do his and only vary the angles
            azim = orientation_array[1,kk]
            ax.view_init(elev=elev, azim=azim)
            
            #Optionally save the figures:
            if save_plot == True:
                self.save_dir = save_dir
                self.base_filename = base_filename
                plt.savefig(os.path.join(self.save_dir,self.base_filename + '__elevation{0}_azimuth{1}.png'.format(elev,azim)))
            
            #If you are saving them as a validation step in part of a script that uses this class, 
            #might want to close figures so don't hog resources (especially if many points):
            if close_plot == True:
                plt.close()
            

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

if __name__ == '__main__':
    
    #==============================================================================
    # Test the DelaunayInterpolator class on some basic 2D data
    #==============================================================================

    #Make a simple Gaussian hill
    Nsamples = 1001
    x = np.linspace(-2.,2.,Nsamples)
    y = np.linspace(-2.,2.,Nsamples)
    xx,yy = np.meshgrid(x,y)
    sigma_x = .3
    sigma_y = .7    
    zz = np.exp(-.5*(xx/sigma_x)**2)*np.exp(-.5*(yy/sigma_y)**2)
    plt.figure()
    #Ymin and ymax are switched in extent=[...] so y axis corresponds to the imshow in later lines
    #in particular, compare to figure with azimuth=270.
    plt.imshow(zz,interpolation='none',extent=[x.min(),x.max(),y.max(),y.min()])
    plt.colorbar()
    
    #Randomly pick a subset of those points
    np.random.seed(87123)
    inds = np.random.choice(np.arange(zz.size),size=Nsamples,replace=False)
    x_sampled = xx.flatten()[inds]
    y_sampled = yy.flatten()[inds]
    z_sampled = zz.flatten()[inds]
    
    #The true data
    data = np.array([x_sampled,y_sampled,z_sampled,]).T
    
    #Make the Delaunay Triangulated surface
    DelaunaySurface = DelaunayInterpolator(data)
    
    #Do a helicopter blade sampling:    
    xy_pairs1 = np.array([ np.linspace(-3.5,1.5,10),np.linspace(-1.5,1.5,10) ]).T#Use -3.5 to give some points outside range so that can see that it treats them as NaNs
    xy_pairs2 = np.array([ np.linspace(-1.5,1.5,10),np.linspace(1.5,-1.5,10) ]).T
    xy_pairs = np.vstack((xy_pairs1,xy_pairs2))
    interpolated_z = DelaunaySurface.interpolate(xy_pairs)
    
    
    #Visualize results
    DelaunaySurface.plot3D(lifted_height=1.,close_plot=False,save_plot=True,save_dir=os.getcwd(),base_filename='Example_Gaussian')
    
    #Print the interpolated values:
    print DelaunaySurface.interpolated_Z_array
    
    
    
    
    
    #==============================================================================
    # Use Delaunay Inteprolation for Image Reconstruction
    #==============================================================================

    import matplotlib.image as mpimg
    filepath = r"Lichtenstein_img_processing_test.png"
    full_image = mpimg.imread(filepath)[:,:,0] #Just use single channel of RGB #.mean(axis=2) instead for greyscale
    
    #Use 10% of the pixels
    Nsamples = round(full_image.size/100.)
    
    x = np.arange(full_image.shape[0])
    y = np.arange(full_image.shape[1])
    xx,yy = np.meshgrid(x,y)
        
    #Randomly pick a subset of those points
    np.random.seed(2398)
    inds = np.random.choice(np.arange(full_image.size),size=Nsamples,replace=False)
    x_sampled = xx.flatten()[inds]
    y_sampled = yy.flatten()[inds]
    z_sampled = full_image.flatten()[inds]
    
    #The true data
    data = np.array([x_sampled,y_sampled,z_sampled,]).T
    
    #Make the Delaunay Triangulated surface
    DelaunaySurface = DelaunayInterpolator(data)
    
    #Want to reconstruct the image by interpolating at every point (including the sampled points)
    xy_pairs = np.vstack((xx.flatten(),yy.flatten())).T
    
#    DelaunaySurface.plot3D(lifted_height=1.,close_plot=False,save_plot=False)
    
    interpolated_z = DelaunaySurface.interpolate(xy_pairs)
    
    
    #Plot the original image, the sample points, and the interpolated image
    fig = plt.figure(figsize=(18,14))
    
    sub1 = fig.add_subplot(131)
    plt.title('Original Image',fontsize=20)
    plt.imshow(full_image,interpolation='none',cmap=plt.cm.Spectral)

    sub2 = fig.add_subplot(132)
    plt.title('Sparsely Sampled (10% of pixels)',fontsize=20)
    zsampled_image = np.nan*np.ones(full_image.size)
    zsampled_image[inds] = full_image.flatten()[inds]
    zsampled_image = zsampled_image.reshape(full_image.shape)
    plt.imshow(zsampled_image,interpolation='none',cmap=plt.cm.Spectral)
    
    sub3 = fig.add_subplot(133)
    plt.title('Reconstructed Image',fontsize=20)
    plt.imshow(interpolated_z.reshape(full_image.shape),interpolation='none',cmap=plt.cm.Spectral)
    
    plt.savefig('Lichtenstein_img_processing_test__RECONSTRUCTED.png')