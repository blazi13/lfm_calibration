import numpy
import math
import time
import scipy.signal
import scipy.ndimage
import scipy.misc
from numpy.fft import fft,ifft,fftshift,fft2
from scipy.linalg import lstsq
from PIL import Image
from zernike import zernike_grad as zernike_grad

class coords:
    def __init__(self,results):
        self.poly_m=None
        self.poly_grid=None
        self.delta_grid=None
        self.xy=results[:,:2]
        self.l=len(self.xy)
        self.xy_image=results[:,2:]
        self.diff_x=numpy.subtract(results[:,0],results[:,2])
        self.diff_y=numpy.subtract(results[:,1],results[:,3])
        self.xy_orig=numpy.add([-screen_range/2,-screen_range/2],self.xy)
        self.xy_orig_image=numpy.add([-screen_range/2,-screen_range/2],self.xy_image)
        subgrid_array=numpy.arange(0,screen_range+subgrid_pitch,subgrid_pitch)
        X,Y=numpy.meshgrid(subgrid_array,subgrid_array)
        self.subgridX=X.ravel()
        self.subgridY=Y.ravel()
        self.subgrid=zip(self.subgridX,self.subgridY)
        self.subgrid_coords=numpy.add([-screen_range/2,-screen_range/2],self.subgrid)
        self.delta=numpy.subtract(self.xy_orig,self.xy_orig_image)
        self.rhophi_image=map(coords.cart2pol,self.xy_orig_image)
        self.polynomials=[]
    def get_std(self,series):
        std=numpy.sqrt(numpy.mean(numpy.multiply(series,series)))
        return(std)
    def show_std(self):
        print "x-deviation:",self.get_std(self.diff_x)
        print "y-deviation:",self.get_std(self.diff_y)
    def get_diff_std(self,xy,xyp):
        diff=numpy.subtract(xy,xyp)
        diff_2=numpy.multiply(diff,diff)
        out=numpy.sqrt(numpy.mean(numpy.sum(diff_2,axis=1)))
        return(out)
    def generate_poly(self,degree):
        out=[]
        z_n=zernike_grad(degree)
        for line in self.xy_orig_image:
            out.append(z_n(*line))
        self.poly=numpy.array(out)
        return(out)
    def generate_poly_multi(self,degree,support):
        t_0=time.time()
        out=[]
        self.degree=degree
        for i in range(1,degree+1):
            print "generating %d-th polynomial"%i
            val=[]
            z_n=zernike_grad(i)
            for line in support:
                val.append(z_n(*line))
            out.append(val)
        return(numpy.array(out))
        
    def fit_correction(self):
 
        d_x0=self.delta[:,0]
        d_x1=self.delta[:,1]
        p_0=self.poly[:,0]
        p_1=self.poly[:,1]
        dx=numpy.concatenate([d_x0,d_x1],axis=0)
        p=numpy.concatenate([p_0,p_1],axis=0)
        lambda_k=(numpy.sum(numpy.multiply(d_x0,p_0))+numpy.sum(numpy.multiply(d_x1,p_1)))/(numpy.sum(numpy.multiply(p_0,p_0))+numpy.sum(numpy.multiply(p_1,p_1)))
        x=numpy.array([p]).transpose()
        #print lstsq(x,dx)[0],lambda_k
        #print "l:=",lambda_k
        #print "err-corr:",numpy.sqrt((1.0/self.l)*numpy.sum(numpy.power(numpy.subtract(dx,numpy.multiply(p,lambda_k)),2)))
        #print "err-not corr:",numpy.sqrt((1.0/self.l)*numpy.sum(numpy.power(dx,2)))
        #print "check0:",numpy.power(dx,2)
        return(lambda_k)
        #return(-lstsq(x,dx)[0][0])
    def fit_correction_multi(self):
 
        d_x0=self.delta[:,0]
        d_x1=self.delta[:,1]
        dx=numpy.concatenate([d_x0,d_x1],axis=0)
        y=[]
        for dp in range(self.degree):
            p_0=self.poly_m[dp][:,0]
            p_1=self.poly_m[dp][:,1]
            y.append(numpy.concatenate([p_0,p_1],axis=0))
        y=numpy.array(y).transpose()
        self.fit_results=lstsq(y,dx)[0]
        #print "fit_results=",self.fit_results
        self.corrected_delta=numpy.einsum("i,ijk",self.fit_results,self.poly_m)
    
    
    def fit_correction_multi2(self):
 
        d_x0=self.delta[:,0]
        d_x1=self.delta[:,1]
        x_corr=[]
        y_corr=[]
        for dp in range(self.degree):
            p_0=self.poly_m[dp][:,0]
            p_1=self.poly_m[dp][:,1]
            x_corr.append(p_0)
            y_corr.append(p_1)
        x_corr=numpy.array(x_corr).transpose()
        y_corr=numpy.array(y_corr).transpose()
        self.fit_results_x=lstsq(x_corr,d_x0)[0]
        self.fit_results_y=lstsq(y_corr,d_x1)[0]
        #print "fit_results_x=",self.fit_results_x
        #print "fit_results_y=",self.fit_results_y
        corrected_delta_x=numpy.einsum("i,ijk",self.fit_results_x,self.poly_m)[:,0]
        corrected_delta_y=numpy.einsum("i,ijk",self.fit_results_y,self.poly_m)[:,1]
        self.corrected_delta=numpy.dstack((corrected_delta_x,corrected_delta_y))[0]
        
        self.grid_delta_x=numpy.einsum("i,ijk",self.fit_results_x,self.poly_grid)[:,0]
        self.grid_delta_y=numpy.einsum("i,ijk",self.fit_results_y,self.poly_grid)[:,1]
        
        self.delta_grid=numpy.dstack((self.grid_delta_x,self.grid_delta_y))[0]
        #print self.corrected_delta.shape
       
        #print "err-corr:",numpy.sqrt((1.0/self.l)*numpy.sum(numpy.power(numpy.subtract(dx,numpy.multiply(p,lambda_k)),2)))
        #print "err-not corr:",numpy.sqrt((1.0/self.l)*numpy.sum(numpy.power(dx,2)))
        #print "check0:",numpy.power(dx,2)
        #return(lambda_k)
        #return(-lstsq(x,dx)[0][0])
    def make_fit_functions(self):
        self.x_fit=scipy.interpolate.interp2d(self.subgridX,self.subgridY,self.grid_delta_x)
        self.y_fit=scipy.interpolate.interp2d(self.subgridX,self.subgridY,self.grid_delta_y)
    def read_fit_functions(self,coords):
        return (coords[0]-self.x_fit(*coords)[0],coords[1]-self.y_fit(*coords)[0])
    @staticmethod
    def cart2pol(xy):
        x=xy[0]
        y=xy[1]
        rho = numpy.sqrt(x**2 + y**2)
        phi = numpy.arctan2(y, x)
        return([rho, phi])
    @staticmethod
    def pol2cart(rho, phi):
        x = rho * numpy.cos(phi)
        y = rho * numpy.sin(phi)
        return(x, y)
    @staticmethod
    def chebyshev_poly(n):
        a=numpy.zeros(30)
        a[n]=1
        f=lambda x,y:numpy.polynomial.chebyshev.chebval([x,y],a)
        return (f)
    @staticmethod
    def open_image(f_name):
        im = Image.open(f_name)
        imdata=numpy.array(im.getdata(),dtype=numpy.uint16)
        imdata=imdata.reshape(im.size[0],im.size[1])
        return(imdata)

#imdata = coords.open_image("C:\\Users\\mysz\\Desktop\\lightfield\\testy_aligment\\t2\\hcal.tif")
imdata = coords.open_image("C:\\Users\\mysz\\Desktop\\lightfield\\calibration1.tif")
#imdata = coords.open_image("E:\\cal.tif")
im1_data =coords.open_image("C:\\Users\\mysz\\Desktop\\lightfield\\testy_aligment\\t2\\u4.tif")
#im1_data = coords.open_image("E:\\1.tif")
output_path = "c:\\tmp\\out.tif"
subgrid_pitch=10.0

iet=scipy.signal.fftconvolve(imdata,imdata[::-1,::-1],mode='same')
data_out=numpy.unravel_index(numpy.argmax(iet),imdata.shape)
print data_out
#iet=scipy.signal.correlate2d(imdata,imdata,mode='same')
print "2"
print iet.shape
freq = numpy.fft.fftshift(numpy.fft.fftfreq(iet.shape[-1]))[:1000]
print 1.0/freq[110]
import matplotlib.pyplot as plt
import matplotlib.cm as cm
fft_iet=numpy.abs(numpy.fft.fftshift(fft2(imdata)))[:1000,:1000]
c_max=10
data_out=numpy.unravel_index(numpy.argmax(fft_iet),fft_iet.shape)
print fft_iet[data_out[0],data_out[1]]
l0=-1.0/freq[data_out[0]]
l1=-1.0/freq[data_out[1]]
rotation=45.0-(360/(2*math.pi))*numpy.arctan(l0/l1)
print "angle=",rotation
rotation_rad=rotation*2*math.pi/360
#plt.imshow(fft_iet)
fig0 = plt.figure(figsize=[10,10])
ax0 = fig0.add_subplot(111) 
ax0.imshow(numpy.multiply(10.0,imdata), cmap = cm.Greys_r)
rotated_image=scipy.ndimage.interpolation.rotate(imdata, rotation, axes=(1, 0), reshape=False)
iet_rotated=scipy.signal.fftconvolve(rotated_image,rotated_image[::-1,::-1],mode='same')
fft_iet_rot=numpy.fft.fftshift(fft2(rotated_image))[:1000,:1000]
data_out=numpy.unravel_index(numpy.argmax(numpy.abs(fft_iet_rot)),fft_iet.shape)
l0_rot=-1.0/freq[data_out[0]]
l1_rot=-1.0/freq[data_out[1]]
pitch_pix=0.5*(l0_rot+l1_rot)
print "pitch_in_pix=",pitch_pix
print pitch_pix*math.sin(rotation_rad),pitch_pix*math.cos(rotation_rad)
def microlens(r2):
    return(numpy.exp(-50*r2/(microlens_rad2)))
    
def draw_gauss(matrix,position,sigma=2):
    s=sigma*6
    sigma_2=sigma**2
    for dx in range(-s,s):
        for dy in range(-s,s):
            x=int(position[0]+dx)
            y=int(position[1]+dy)
            dxx=x-position[0]
            dyy=y-position[1]
            matrix[x,y]=numpy.exp(-(dxx**2+dyy**2)/sigma_2)
            

def make_shift(x,y,pitch):
	x_i=math.floor(x/pitch)
	y_i=math.floor(y/pitch)
	return ([x-pitch*(x_i+0.5),y-pitch*(y_i+0.5)])

def real_position(x,y,z):
        x_pos=(x-xy_range/2)*grid_xy_fact
        y_pos=(y-xy_range/2)*grid_xy_fact
        z_pos=z*grid_z_fact
        return ([x_pos,y_pos,z_pos])

def screen_position(x,y):
        x_pos=(x-screen_range/2)*screen_xy_fact
        y_pos=(y-screen_range/2)*screen_xy_fact
        return ([x_pos,y_pos])
screen_range=imdata.shape[0]
screen_dim=120e-4
pix_size=screen_dim/screen_range

microlens_pitch=pitch_pix*pix_size
screen_xy_fact=screen_dim/screen_range
microlens_rad2=microlens_pitch**2/4.0
l=525e-9
n=1
k=2*numpy.pi*n/l
microlens_matrix=numpy.zeros([screen_range,screen_range],dtype=numpy.float)
for x in range(screen_range):
    print round(100.0*x/screen_range,3)," % done"
    for y in range(screen_range):
        x1,x2=screen_position(x,y)
        x_m1,x_m2=make_shift(x1,x2,microlens_pitch)
        r2=x_m1**2+x_m2**2
        #print x,y,x_m1,x_m2
        #print microlens(0),microlens(microlens_rad2/2.0)
        if r2<microlens_rad2:
            microlens_matrix[x,y]=microlens(r2)
fig1 = plt.figure(figsize=[10,10])
ax1 = fig1.add_subplot(111) 
fig2 = plt.figure(figsize=[10,10])
ax2 = fig2.add_subplot(111) 
microlens_matrix=numpy.multiply(1.0/numpy.max(microlens_matrix),microlens_matrix)
rotated_image=numpy.multiply(1.0/numpy.max(rotated_image),rotated_image)
ax1.imshow(numpy.abs(rotated_image), cmap = cm.Greys_r)
ax2.imshow(numpy.dstack((numpy.abs(microlens_matrix),rotated_image,numpy.zeros([screen_range,screen_range]))))
iet_shift=scipy.signal.fftconvolve(microlens_matrix,rotated_image[::-1,::-1],mode='same')
data_out=numpy.unravel_index(numpy.argmax(numpy.abs(iet_shift)),iet_shift.shape)
shift=numpy.subtract(data_out,numpy.multiply(0.5,iet_shift.shape))
shift_image=scipy.ndimage.interpolation.shift(rotated_image,shift) 
shift_image=numpy.multiply(1.0/numpy.max(shift_image),shift_image)
fig3 = plt.figure(figsize=[40,40])
max_shift_image=0.1*numpy.max(shift_image)
for i in range(screen_range):
    for j in range(screen_range):
        if shift_image[i,j]<max_shift_image:
            shift_image[i,j]=0
print "shift=",shift
pitch_pix_nr=int(screen_range/pitch_pix)
pitch_pix_int=int(pitch_pix)
pitch_pix_int_2=int(pitch_pix)/2
microlens_position=numpy.zeros([screen_range,screen_range])
cn=numpy.zeros([screen_range,screen_range])
results=[]
for x_i in range(-pitch_pix_nr/2,pitch_pix_nr/2-1):
    for y_i in range(-pitch_pix_nr/2,pitch_pix_nr/2-1):
        x=int(round((0.5+x_i)*pitch_pix,0))+screen_range/2
        y=int(round((0.5+y_i)*pitch_pix,0))+screen_range/2
        x_l=x-pitch_pix_int_2
        x_r=x+pitch_pix_int_2
        y_l=y-pitch_pix_int_2
        y_r=y+pitch_pix_int_2
        #print x_l,x_r,y_l,y_r
        
        img_cr=shift_image[x_l:x_r,y_l:y_r]
        mlens_cr=microlens_matrix[x_l:x_r,y_l:y_r]
        im_cm=scipy.ndimage.measurements.center_of_mass(img_cr)
        #print x_i,y_i,x_l,y_l,x_r,y_r,im_cm
        m_lens_cm=scipy.ndimage.measurements.center_of_mass(mlens_cr)
        try:
            microlens_position[int(round(x_l+im_cm[0],0)),int(round(y_l+im_cm[1],0))]=1
            cn[int(x_l),int(y_l)]=1
            results.append([x_l+m_lens_cm[0],y_l+m_lens_cm[1],x_l+im_cm[0],y_l+im_cm[1]])
        except(ValueError,IndexError):
            pass
#standard_dev
results=numpy.array(results)


from scipy import ndimage
import scipy.misc
scipy.misc.toimage(microlens_matrix).save('microlens_matrix.tif')
ax3 = fig3.add_subplot(111) 
ax3.imshow(numpy.dstack((microlens_matrix,shift_image,numpy.zeros([screen_range,screen_range]))))
pcr=coords(results)
pcr.show_std()
print "std_0=",pcr.get_diff_std(pcr.xy_orig,pcr.xy_orig_image)
#for nr in range(1,10):
#    pcr.generate_poly(nr)
#    cf=pcr.fit_correction()
#    print nr,"std_1=",pcr.get_diff_std(pcr.xy_orig,numpy.add(numpy.multiply(cf,pcr.poly),pcr.xy_orig_image))

#for k in range(1,80,1):
#    pcr.poly_m=pcr.generate_poly_multi(k,pcr.xy_orig_image)
#    pcr.poly_grid=pcr.generate_poly_multi(k,pcr.subgrid_coords)
#    cf1=pcr.fit_correction_multi2()
#    print k,"std_m=",pcr.get_diff_std(pcr.xy_orig,numpy.add(pcr.corrected_delta,pcr.xy_orig_image))
#os._exit(0)	
pcr.poly_m=pcr.generate_poly_multi(15,pcr.xy_orig_image)
pcr.poly_grid=pcr.generate_poly_multi(15,pcr.subgrid_coords)
cf1=pcr.fit_correction_multi2()
print k,"std_m=",pcr.get_diff_std(pcr.xy_orig,numpy.add(pcr.corrected_delta,pcr.xy_orig_image))

pcr.make_fit_functions()

fig4 = plt.figure(figsize=[40,40])
ax4 = fig4.add_subplot(111)
c_corrected=numpy.zeros([screen_range,screen_range],dtype=numpy.float)
for point in numpy.add(pcr.corrected_delta,pcr.xy_image):
    draw_gauss(c_corrected,point,sigma=1)

transformed_image=ndimage.geometric_transform(shift_image, pcr.read_fit_functions,order=1)    

ax4.imshow(numpy.dstack((microlens_matrix,transformed_image,numpy.zeros([screen_range,screen_range]))))
im1_data=scipy.ndimage.interpolation.rotate(im1_data, rotation, axes=(1, 0), reshape=False)
im1_data=scipy.ndimage.interpolation.shift(im1_data,shift)
fig3.savefig("figure1.tif")
fig4.savefig("figure2.tif")
im1_transformed=ndimage.geometric_transform(im1_data, pcr.read_fit_functions,order=1)

scipy.misc.toimage(im1_transformed).save(output_path)
plt.show()
