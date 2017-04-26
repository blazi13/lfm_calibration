import numpy
from scipy.special import binom
from math import factorial,ceil,sqrt
import csv

line_parser=lambda x: [int(x[0]),int(x[1]),int(x[2]),str(x[3])]
x_der_file=map(line_parser,list(csv.reader(open("xder.txt"))))
y_der_file=map(line_parser,list(csv.reader(open("yder.txt"))))

def zernike_grad(n):
    x_s=x_der_file[n][3]
    y_s=y_der_file[n][3]
    f=lambda x,y:[float(eval(x_s)),float(eval(y_s))]
    fp=lambda x,y: f(x/1000.0,y/1000.0)
    return(fp)
 

def zernike_grad1(n):
    def f(x,y):
        xp=x/1000.0
        yp=y/1000.0
        r=numpy.sqrt((xp)**2+(yp)**2)
        xd=xp/r
        yd=yp/r
        return([xd*r**n,yd*r**n])
    return(f)
 
 
def zernike_grad3(n): 
    if n==0:
             f=lambda x,y:[0,0]
    if n==1:
             f=lambda x,y:[1,0]
    if n==2:
             f=lambda x,y:[0,1]
    if n==3:
             f=lambda x,y:[4*x,4*y]
    if n==4:
             f=lambda x,y:[2*x,-2*y]
    if n==5:
             f=lambda x,y:[2*y,2*x]
    if n==6:
             f=lambda x,y:[-2+9*x**2+3*y**2,6*x*y]
    if n==7:
             f=lambda x,y:[6*x*y,-2+3*x**2+9*y**2]
    if n==8:
             f=lambda x,y:[-12*x+24*x**3+24*x*y**2,-12*y+24*x**2*y+24*y**3]
    if n==9:
             f=lambda x,y:[3*x**2-3*y**2,-6*x*y]
    if n==10:
             f=lambda x,y:[6*x*y,3*x**2-3*y**2]
    if n==11:
             f=lambda x,y:[-6*x+16*x**3,6*y-16*y**3]
    if n==12:
             f=lambda x,y:[-6*y+24*x**2*y+8*y**3,-6*x+8*x**3+24*x*y**2]
    if n==13:
             f=lambda x,y:[3-36*x**2+50*x**4-12*y**2+60*x**2*y**2+10*y**4,-24*x*y+40*x**3*y+40*x*y**3]
    if n==14:
             f=lambda x,y:[-24*x*y+40*x**3*y+40*x*y**3,3-12*x**2+10*x**4-36*y**2+60*x**2*y**2+50*y**4]
    if n==15:
             f=lambda x,y:[24*x-120*x**3+120*x**5-120*x*y**2+240*x**3*y**2+120*x*y**4,24*y-120*x**2*y+120*x**4*y-120*y**3+240*x**2*y**3+120*y**5]
    if n==16:
             f=lambda x,y:[4*x**3-12*x*y**2,-12*x**2*y+4*y**3]
    if n==17:
             f=lambda x,y:[12*x**2*y-4*y**3,4*x**3-12*x*y**2]
    if n==18:
             f=lambda x,y:[-12*x**2+25*x**4+12*y**2-30*x**2*y**2-15*y**4,24*x*y-20*x**3*y-60*x*y**3]
    if n==19:
             f=lambda x,y:[-24*x*y+60*x**3*y+20*x*y**3,-12*x**2+15*x**4+12*y**2+30*x**2*y**2-25*y**4]
    if n==20:
             f=lambda x,y:[12*x-80*x**3+90*x**5+60*x**3*y**2-30*x*y**4,-12*y+30*x**4*y+80*y**3-60*x**2*y**3-90*y**5]
    if n==21:
             f=lambda x,y:[12*y-120*x**2*y+150*x**4*y-40*y**3+180*x**2*y**3+30*y**5,12*x-40*x**3+30*x**5-120*x*y**2+180*x**3*y**2+150*x*y**4]
    if n==22:
             f=lambda x,y:[-4+90*x**2-300*x**4+245*x**6+30*y**2-360*x**2*y**2+525*x**4*y**2-60*y**4+315*x**2*y**4+35*y**6,60*x*y-240*x**3*y+210*x**5*y-240*x*y**3+420*x**3*y**3+210*x*y**5]
    if n==23:
             f=lambda x,y:[60*x*y-240*x**3*y+210*x**5*y-240*x*y**3+420*x**3*y**3+210*x*y**5,-4+30*x**2-60*x**4+35*x**6+90*y**2-360*x**2*y**2+315*x**4*y**2-300*y**4+525*x**2*y**4+245*y**6]
    if n==24:
             f=lambda x,y:[-40*x+360*x**3-840*x**5+560*x**7+360*x*y**2-1680*x**3*y**2+1680*x**5*y**2-840*x*y**4+1680*x**3*y**4+560*x*y**6,-40*y+360*x**2*y-840*x**4*y+560*x**6*y+360*y**3-1680*x**2*y**3+1680*x**4*y**3-840*y**5+1680*x**2*y**5+560*y**7]
    if n==25:
             f=lambda x,y:[5*x**4-30*x**2*y**2+5*y**4,-20*x**3*y+20*x*y**3]
    if n==26:
             f=lambda x,y:[20*x**3*y-20*x*y**3,5*x**4-30*x**2*y**2+5*y**4]
    if n==27:
             f=lambda x,y:[-20*x**3+36*x**5+60*x*y**2-120*x**3*y**2-60*x*y**4,60*x**2*y-60*x**4*y-20*y**3-120*x**2*y**3+36*y**5]
    if n==28:
             f=lambda x,y:[-60*x**2*y+120*x**4*y+20*y**3-24*y**5,-20*x**3+24*x**5+60*x*y**2-120*x*y**4]
    if n==29:
             f=lambda x,y:[30*x**2-150*x**4+147*x**6-30*y**2+180*x**2*y**2-105*x**4*y**2+90*y**4-315*x**2*y**4-63*y**6,-60*x*y+120*x**3*y-42*x**5*y+360*x*y**3-420*x**3*y**3-378*x*y**5]
    if n==30:
             f=lambda x,y:[60*x*y-360*x**3*y+378*x**5*y-120*x*y**3+420*x**3*y**3+42*x*y**5,30*x**2-90*x**4+63*x**6-30*y**2-180*x**2*y**2+315*x**4*y**2+150*y**4+105*x**2*y**4-147*y**6]
    if n==31:
             f=lambda x,y:[-20*x+240*x**3-630*x**5+448*x**7-420*x**3*y**2+672*x**5*y**2+210*x*y**4-224*x*y**6,20*y-210*x**4*y+224*x**6*y-240*y**3+420*x**2*y**3+630*y**5-672*x**2*y**5-448*y**7]
    if n==32:
             f=lambda x,y:[-20*y+360*x**2*y-1050*x**4*y+784*x**6*y+120*y**3-1260*x**2*y**3+1680*x**4*y**3-210*y**5+1008*x**2*y**5+112*y**7,-20*x+120*x**3-210*x**5+112*x**7+360*x*y**2-1260*x**3*y**2+1008*x**5*y**2-1050*x*y**4+1680*x**3*y**4+784*x*y**6]
    if n==33:
             f=lambda x,y:[5-180*x**2+1050*x**4-1960*x**6+1134*x**8-60*y**2+1260*x**2*y**2-4200*x**4*y**2+3528*x**6*y**2+210*y**4-2520*x**2*y**4+3780*x**4*y**4-280*y**6+1512*x**2*y**6+126*y**8,-120*x*y+840*x**3*y-1680*x**5*y+1008*x**7*y+840*x*y**3-3360*x**3*y**3+3024*x**5*y**3-1680*x*y**5+3024*x**3*y**5+1008*x*y**7]
    if n==34:
             f=lambda x,y:[-120*x*y+840*x**3*y-1680*x**5*y+1008*x**7*y+840*x*y**3-3360*x**3*y**3+3024*x**5*y**3-1680*x*y**5+3024*x**3*y**5+1008*x*y**7,5-60*x**2+210*x**4-280*x**6+126*x**8-180*y**2+1260*x**2*y**2-2520*x**4*y**2+1512*x**6*y**2+1050*y**4-4200*x**2*y**4+3780*x**4*y**4-1960*y**6+3528*x**2*y**6+1134*y**8]
    if n==35:
             f=lambda x,y:[60*x-840*x**3+3360*x**5-5040*x**7+2520*x**9-840*x*y**2+6720*x**3*y**2-15120*x**5*y**2+10080*x**7*y**2+3360*x*y**4-15120*x**3*y**4+15120*x**5*y**4-5040*x*y**6+10080*x**3*y**6+2520*x*y**8,60*y-840*x**2*y+3360*x**4*y-5040*x**6*y+2520*x**8*y-840*y**3+6720*x**2*y**3-15120*x**4*y**3+10080*x**6*y**3+3360*y**5-15120*x**2*y**5+15120*x**4*y**5-5040*y**7+10080*x**2*y**7+2520*y**9]
    if n==36:
             f=lambda x,y:[6*x**5-60*x**3*y**2+30*x*y**4,-30*x**4*y+60*x**2*y**3-6*y**5]
    if n==37:
             f=lambda x,y:[30*x**4*y-60*x**2*y**3+6*y**5,6*x**5-60*x**3*y**2+30*x*y**4]
    if n==38:
             f=lambda x,y:[-30*x**4+49*x**6+180*x**2*y**2-315*x**4*y**2-30*y**4-105*x**2*y**4+35*y**6,120*x**3*y-126*x**5*y-120*x*y**3-140*x**3*y**3+210*x*y**5]
    if n==39:
             f=lambda x,y:[-120*x**3*y+210*x**5*y+120*x*y**3-140*x**3*y**3-126*x*y**5,-30*x**4+35*x**6+180*x**2*y**2-105*x**4*y**2-30*y**4-315*x**2*y**4+49*y**6]
    if n==40:
             f=lambda x,y:[60*x**3-252*x**5+224*x**7-180*x*y**2+840*x**3*y**2-672*x**5*y**2+420*x*y**4-1120*x**3*y**4-224*x*y**6,-180*x**2*y+420*x**4*y-224*x**6*y+60*y**3+840*x**2*y**3-1120*x**4*y**3-252*y**5-672*x**2*y**5+224*y**7]
    if n==41:
             f=lambda x,y:[180*x**2*y-840*x**4*y+784*x**6*y-60*y**3+560*x**4*y**3+168*y**5-336*x**2*y**5-112*y**7,60*x**3-168*x**5+112*x**7-180*x*y**2+336*x**5*y**2+840*x*y**4-560*x**3*y**4-784*x*y**6]
    if n==42:
             f=lambda x,y:[-60*x**2+525*x**4-1176*x**6+756*x**8+60*y**2-630*x**2*y**2+840*x**4*y**2-315*y**4+2520*x**2*y**4-2520*x**4*y**4+504*y**6-2016*x**2*y**6-252*y**8,120*x*y-420*x**3*y+336*x**5*y-1260*x*y**3+3360*x**3*y**3-2016*x**5*y**3+3024*x*y**5-4032*x**3*y**5-2016*x*y**7]
    if n==43:
             f=lambda x,y:[-120*x*y+1260*x**3*y-3024*x**5*y+2016*x**7*y+420*x*y**3-3360*x**3*y**3+4032*x**5*y**3-336*x*y**5+2016*x**3*y**5,-60*x**2+315*x**4-504*x**6+252*x**8+60*y**2+630*x**2*y**2-2520*x**4*y**2+2016*x**6*y**2-525*y**4-840*x**2*y**4+2520*x**4*y**4+1176*y**6-756*y**8]
    if n==44:
             f=lambda x,y:[30*x-560*x**3+2520*x**5-4032*x**7+2100*x**9+1680*x**3*y**2-6048*x**5*y**2+5040*x**7*y**2-840*x*y**4+2520*x**5*y**4+2016*x*y**6-1680*x**3*y**6-1260*x*y**8,-30*y+840*x**4*y-2016*x**6*y+1260*x**8*y+560*y**3-1680*x**2*y**3+1680*x**6*y**3-2520*y**5+6048*x**2*y**5-2520*x**4*y**5+4032*y**7-5040*x**2*y**7-2100*y**9]
    if n==45:
             f=lambda x,y:[30*y-840*x**2*y+4200*x**4*y-7056*x**6*y+3780*x**8*y-280*y**3+5040*x**2*y**3-15120*x**4*y**3+11760*x**6*y**3+840*y**5-9072*x**2*y**5+12600*x**4*y**5-1008*y**7+5040*x**2*y**7+420*y**9,30*x-280*x**3+840*x**5-1008*x**7+420*x**9-840*x*y**2+5040*x**3*y**2-9072*x**5*y**2+5040*x**7*y**2+4200*x*y**4-15120*x**3*y**4+12600*x**5*y**4-7056*x*y**6+11760*x**3*y**6+3780*x*y**8]
    if n==46:
             f=lambda x,y:[-6+315*x**2-2800*x**4+8820*x**6-11340*x**8+5082*x**10+105*y**2-3360*x**2*y**2+18900*x**4*y**2-35280*x**6*y**2+20790*x**8*y**2-560*y**4+11340*x**2*y**4-37800*x**4*y**4+32340*x**6*y**4+1260*y**6-15120*x**2*y**6+23100*x**4*y**6-1260*y**8+6930*x**2*y**8+462*y**10,210*x*y-2240*x**3*y+7560*x**5*y-10080*x**7*y+4620*x**9*y-2240*x*y**3+15120*x**3*y**3-30240*x**5*y**3+18480*x**7*y**3+7560*x*y**5-30240*x**3*y**5+27720*x**5*y**5-10080*x*y**7+18480*x**3*y**7+4620*x*y**9]
    if n==47:
             f=lambda x,y:[210*x*y-2240*x**3*y+7560*x**5*y-10080*x**7*y+4620*x**9*y-2240*x*y**3+15120*x**3*y**3-30240*x**5*y**3+18480*x**7*y**3+7560*x*y**5-30240*x**3*y**5+27720*x**5*y**5-10080*x*y**7+18480*x**3*y**7+4620*x*y**9,-6+105*x**2-560*x**4+1260*x**6-1260*x**8+462*x**10+315*y**2-3360*x**2*y**2+11340*x**4*y**2-15120*x**6*y**2+6930*x**8*y**2-2800*y**4+18900*x**2*y**4-37800*x**4*y**4+23100*x**6*y**4+8820*y**6-35280*x**2*y**6+32340*x**4*y**6-11340*y**8+20790*x**2*y**8+5082*y**10]
    if n==48:
             f=lambda x,y:[-84*x+1680*x**3-10080*x**5+25200*x**7-27720*x**9+11088*x**11+1680*x*y**2-20160*x**3*y**2+75600*x**5*y**2-110880*x**7*y**2+55440*x**9*y**2-10080*x*y**4+75600*x**3*y**4-166320*x**5*y**4+110880*x**7*y**4+25200*x*y**6-110880*x**3*y**6+110880*x**5*y**6-27720*x*y**8+55440*x**3*y**8+11088*x*y**10,-84*y+1680*x**2*y-10080*x**4*y+25200*x**6*y-27720*x**8*y+11088*x**10*y+1680*y**3-20160*x**2*y**3+75600*x**4*y**3-110880*x**6*y**3+55440*x**8*y**3-10080*y**5+75600*x**2*y**5-166320*x**4*y**5+110880*x**6*y**5+25200*y**7-110880*x**2*y**7+110880*x**4*y**7-27720*y**9+55440*x**2*y**9+11088*y**11]
    if n==49:
             f=lambda x,y:[7*x**6-105*x**4*y**2+105*x**2*y**4-7*y**6,-42*x**5*y+140*x**3*y**3-42*x*y**5]
    fp=lambda x,y: f(x/1000.0,y/1000.0)
    return(fp)

def zernike_grad3(n):
    if n==0:
        c=[0,0]
        f=lambda x,y:[0,0]
    if n==1:
        c=[1,1]
        f=lambda x,y:[1,0]
    if n==2:
        c=[1,-1]
        f=lambda x,y:[0,1]
    if n==3:
        c=[1,0]
        f=lambda x,y:[4*x,4*y]
    if n==4:
        c=[2,2]
        f=lambda x,y:[2*x,-2*y]
    if n==5:
        c=[2,-2]
        f=lambda x,y:[2*y,2*x]
    if n==6:
        c=[2,1]
        f=lambda x,y:[-2.0+9*x**2+3*y**2,6*x*y]
    if n==7:
        c=[2,-1]
        f=lambda x,y:[6*x*y,-2.0+6*y**2+3*x**2]
    if n==8:
        c=[2,0]
        f=lambda x,y:[-12*x+24*x**3+24*x*y**2,12*y-24*x**3-24*y*x**2]
    if n==9:
        c=[3,3]
        f=lambda x,y:[3*x**2-3*y**2,-6*x*y]
    if n==10:
        c=[3,-3]
        f=lambda x,y:[6*x*y,3*x**2-3*y**2]
    if n==11:
        c=[3,2]
        f=lambda x,y:[-6*x**2+16*x**3,6*y**2-16*y*3]
    if n==12:
        c=[3,-2]
        f=lambda x,y:[-6*y+24*x**2*y+8*y**3,-6*x+24*y**2*y+8*x**3]
    if n==13:
        c=[3,1]
        f=lambda x,y:[3-36*x**2-12*y**2+50*x**4+60*x**2*y**2+10*y**4,-24*x*y+40*x**3*y+40*x*y**3]
    if n==14:
        c=[3,-1]
        f=lambda x,y:[-24*x*y+40*x**3*y+40*x*y**3,3-12*x**2-36*y**2+10*x**4+60*x**2*y**2+50*y**4]
    if n==15:
        c=[3,0]
        f=lambda x,y:[24*x-120*x**3-120*x*y**2+120*x*(x**2+y**2)**2,24*y-120*y**3-120*y*x**2+120*y*(y**2+x**2)**2]
    if n==16:
        c=[4,4]
        f=lambda x,y:[4*x**3-12*x*y**2,-12*x**2*y+4*y**3]
    if n==17:
        c=[4,-4]
        f=lambda x,y:[12*x**2*y-4*y**3,4*x**3-12*x*y**2]
    if n==18:
             f=lambda x,y:[-12*x**2+25*x**4+12*y**2-30*x**2*y**2-15*y**4,24*x*y-20*x**3*y-60*x*y**3]
    if n==19:
             f=lambda x,y:[-24*x*y+60*x**3*y+20*x*y**3,-12*x**2+15*x**4+12*y**2+30*x**2*y**2-25*y**4]
    if n==20:
             f=lambda x,y:[12*x-80*x**3+90*x**5+60*x**3*y**2-30*x*y**4,-12*y+30*x**4*y+80*y**3-60*x**2*y**3-90*y**5]
    if n==21:
             f=lambda x,y:[12*y-120*x**2*y+150*x**4*y-40*y**3+180*x**2*y**3+30*y**5,12*x-40*x**3+30*x**5-120*x*y**2+180*x**3*y**2+150*x*y**4]
    if n==22:
             f=lambda x,y:[-4+90*x**2-300*x**4+245*x**6+30*y**2-360*x**2*y**2+525*x**4*y**2-60*y**4+315*x**2*y**4+35*y**6,60*x*y-240*x**3*y+210*x**5*y-240*x*y**3+420*x**3*y**3+210*x*y**5]
    if n==23:
             f=lambda x,y:[60*x*y-240*x**3*y+210*x**5*y-240*x*y**3+420*x**3*y**3+210*x*y**5,-4+30*x**2-60*x**4+35*x**6+90*y**2-360*x**2*y**2+315*x**4*y**2-300*y**4+525*x**2*y**4+245*y**6]
    if n==24:
             f=lambda x,y:[-40*x+360*x**3-840*x**5+560*x**7+360*x*y**2-1680*x**3*y**2+1680*x**5*y**2-840*x*y**4+1680*x**3*y**4+560*x*y**6,-40*y+360*x**2*y-840*x**4*y+560*x**6*y+360*y**3-1680*x**2*y**3+1680*x**4*y**3-840*y**5+1680*x**2*y**5+560*y**7]
    if n==25:
             f=lambda x,y:[5*x**4-30*x**2*y**2+5*y**4,-20*x**3*y+20*x*y**3]
    if n==26:
             f=lambda x,y:[20*x**3*y-20*x*y**3,5*x**4-30*x**2*y**2+5*y**4]
    if n==27:
             f=lambda x,y:[-20*x**3+36*x**5+60*x*y**2-120*x**3*y**2-60*x*y**4,60*x**2*y-60*x**4*y-20*y**3-120*x**2*y**3+36*y**5]
    if n==28:
             f=lambda x,y:[-60*x**2*y+120*x**4*y+20*y**3-24*y**5,-20*x**3+24*x**5+60*x*y**2-120*x*y**4]
    if n==29:
             f=lambda x,y:[30*x**2-150*x**4+147*x**6-30*y**2+180*x**2*y**2-105*x**4*y**2+90*y**4-315*x**2*y**4-63*y**6,-60*x*y+120*x**3*y-42*x**5*y+360*x*y**3-420*x**3*y**3-378*x*y**5]
    if n==30:
             f=lambda x,y:[60*x*y-360*x**3*y+378*x**5*y-120*x*y**3+420*x**3*y**3+42*x*y**5,30*x**2-90*x**4+63*x**6-30*y**2-180*x**2*y**2+315*x**4*y**2+150*y**4+105*x**2*y**4-147*y**6]
    if n==31:
             f=lambda x,y:[-20*x+240*x**3-630*x**5+448*x**7-420*x**3*y**2+672*x**5*y**2+210*x*y**4-224*x*y**6,20*y-210*x**4*y+224*x**6*y-240*y**3+420*x**2*y**3+630*y**5-672*x**2*y**5-448*y**7]
    if n==32:
             f=lambda x,y:[-20*y+360*x**2*y-1050*x**4*y+784*x**6*y+120*y**3-1260*x**2*y**3+1680*x**4*y**3-210*y**5+1008*x**2*y**5+112*y**7,-20*x+120*x**3-210*x**5+112*x**7+360*x*y**2-1260*x**3*y**2+1008*x**5*y**2-1050*x*y**4+1680*x**3*y**4+784*x*y**6]
    if n==33:
             f=lambda x,y:[5-180*x**2+1050*x**4-1960*x**6+1134*x**8-60*y**2+1260*x**2*y**2-4200*x**4*y**2+3528*x**6*y**2+210*y**4-2520*x**2*y**4+3780*x**4*y**4-280*y**6+1512*x**2*y**6+126*y**8,-120*x*y+840*x**3*y-1680*x**5*y+1008*x**7*y+840*x*y**3-3360*x**3*y**3+3024*x**5*y**3-1680*x*y**5+3024*x**3*y**5+1008*x*y**7]
    if n==34:
             f=lambda x,y:[-120*x*y+840*x**3*y-1680*x**5*y+1008*x**7*y+840*x*y**3-3360*x**3*y**3+3024*x**5*y**3-1680*x*y**5+3024*x**3*y**5+1008*x*y**7,5-60*x**2+210*x**4-280*x**6+126*x**8-180*y**2+1260*x**2*y**2-2520*x**4*y**2+1512*x**6*y**2+1050*y**4-4200*x**2*y**4+3780*x**4*y**4-1960*y**6+3528*x**2*y**6+1134*y**8]
    if n==35:
             f=lambda x,y:[60*x-840*x**3+3360*x**5-5040*x**7+2520*x**9-840*x*y**2+6720*x**3*y**2-15120*x**5*y**2+10080*x**7*y**2+3360*x*y**4-15120*x**3*y**4+15120*x**5*y**4-5040*x*y**6+10080*x**3*y**6+2520*x*y**8,60*y-840*x**2*y+3360*x**4*y-5040*x**6*y+2520*x**8*y-840*y**3+6720*x**2*y**3-15120*x**4*y**3+10080*x**6*y**3+3360*y**5-15120*x**2*y**5+15120*x**4*y**5-5040*y**7+10080*x**2*y**7+2520*y**9]
    if n==36:
             f=lambda x,y:[6*x**5-60*x**3*y**2+30*x*y**4,-30*x**4*y+60*x**2*y**3-6*y**5]
    if n==37:
             f=lambda x,y:[30*x**4*y-60*x**2*y**3+6*y**5,6*x**5-60*x**3*y**2+30*x*y**4]
    if n==38:
             f=lambda x,y:[-30*x**4+49*x**6+180*x**2*y**2-315*x**4*y**2-30*y**4-105*x**2*y**4+35*y**6,120*x**3*y-126*x**5*y-120*x*y**3-140*x**3*y**3+210*x*y**5]
    if n==39:
             f=lambda x,y:[-120*x**3*y+210*x**5*y+120*x*y**3-140*x**3*y**3-126*x*y**5,-30*x**4+35*x**6+180*x**2*y**2-105*x**4*y**2-30*y**4-315*x**2*y**4+49*y**6]
    if n==40:
             f=lambda x,y:[60*x**3-252*x**5+224*x**7-180*x*y**2+840*x**3*y**2-672*x**5*y**2+420*x*y**4-1120*x**3*y**4-224*x*y**6,-180*x**2*y+420*x**4*y-224*x**6*y+60*y**3+840*x**2*y**3-1120*x**4*y**3-252*y**5-672*x**2*y**5+224*y**7]
    if n==41:
             f=lambda x,y:[180*x**2*y-840*x**4*y+784*x**6*y-60*y**3+560*x**4*y**3+168*y**5-336*x**2*y**5-112*y**7,60*x**3-168*x**5+112*x**7-180*x*y**2+336*x**5*y**2+840*x*y**4-560*x**3*y**4-784*x*y**6]
    if n==42:
             f=lambda x,y:[-60*x**2+525*x**4-1176*x**6+756*x**8+60*y**2-630*x**2*y**2+840*x**4*y**2-315*y**4+2520*x**2*y**4-2520*x**4*y**4+504*y**6-2016*x**2*y**6-252*y**8,120*x*y-420*x**3*y+336*x**5*y-1260*x*y**3+3360*x**3*y**3-2016*x**5*y**3+3024*x*y**5-4032*x**3*y**5-2016*x*y**7]
    if n==43:
             f=lambda x,y:[-120*x*y+1260*x**3*y-3024*x**5*y+2016*x**7*y+420*x*y**3-3360*x**3*y**3+4032*x**5*y**3-336*x*y**5+2016*x**3*y**5,-60*x**2+315*x**4-504*x**6+252*x**8+60*y**2+630*x**2*y**2-2520*x**4*y**2+2016*x**6*y**2-525*y**4-840*x**2*y**4+2520*x**4*y**4+1176*y**6-756*y**8]
    if n==44:
             f=lambda x,y:[30*x-560*x**3+2520*x**5-4032*x**7+2100*x**9+1680*x**3*y**2-6048*x**5*y**2+5040*x**7*y**2-840*x*y**4+2520*x**5*y**4+2016*x*y**6-1680*x**3*y**6-1260*x*y**8,-30*y+840*x**4*y-2016*x**6*y+1260*x**8*y+560*y**3-1680*x**2*y**3+1680*x**6*y**3-2520*y**5+6048*x**2*y**5-2520*x**4*y**5+4032*y**7-5040*x**2*y**7-2100*y**9]
    if n==45:
             f=lambda x,y:[30*y-840*x**2*y+4200*x**4*y-7056*x**6*y+3780*x**8*y-280*y**3+5040*x**2*y**3-15120*x**4*y**3+11760*x**6*y**3+840*y**5-9072*x**2*y**5+12600*x**4*y**5-1008*y**7+5040*x**2*y**7+420*y**9,30*x-280*x**3+840*x**5-1008*x**7+420*x**9-840*x*y**2+5040*x**3*y**2-9072*x**5*y**2+5040*x**7*y**2+4200*x*y**4-15120*x**3*y**4+12600*x**5*y**4-7056*x*y**6+11760*x**3*y**6+3780*x*y**8]
    if n==46:
             f=lambda x,y:[-6+315*x**2-2800*x**4+8820*x**6-11340*x**8+5082*x**10+105*y**2-3360*x**2*y**2+18900*x**4*y**2-35280*x**6*y**2+20790*x**8*y**2-560*y**4+11340*x**2*y**4-37800*x**4*y**4+32340*x**6*y**4+1260*y**6-15120*x**2*y**6+23100*x**4*y**6-1260*y**8+6930*x**2*y**8+462*y**10,210*x*y-2240*x**3*y+7560*x**5*y-10080*x**7*y+4620*x**9*y-2240*x*y**3+15120*x**3*y**3-30240*x**5*y**3+18480*x**7*y**3+7560*x*y**5-30240*x**3*y**5+27720*x**5*y**5-10080*x*y**7+18480*x**3*y**7+4620*x*y**9]
    if n==47:
             f=lambda x,y:[210*x*y-2240*x**3*y+7560*x**5*y-10080*x**7*y+4620*x**9*y-2240*x*y**3+15120*x**3*y**3-30240*x**5*y**3+18480*x**7*y**3+7560*x*y**5-30240*x**3*y**5+27720*x**5*y**5-10080*x*y**7+18480*x**3*y**7+4620*x*y**9,-6+105*x**2-560*x**4+1260*x**6-1260*x**8+462*x**10+315*y**2-3360*x**2*y**2+11340*x**4*y**2-15120*x**6*y**2+6930*x**8*y**2-2800*y**4+18900*x**2*y**4-37800*x**4*y**4+23100*x**6*y**4+8820*y**6-35280*x**2*y**6+32340*x**4*y**6-11340*y**8+20790*x**2*y**8+5082*y**10]
    if n==48:
             f=lambda x,y:[-84*x+1680*x**3-10080*x**5+25200*x**7-27720*x**9+11088*x**11+1680*x*y**2-20160*x**3*y**2+75600*x**5*y**2-110880*x**7*y**2+55440*x**9*y**2-10080*x*y**4+75600*x**3*y**4-166320*x**5*y**4+110880*x**7*y**4+25200*x*y**6-110880*x**3*y**6+110880*x**5*y**6-27720*x*y**8+55440*x**3*y**8+11088*x*y**10,-84*y+1680*x**2*y-10080*x**4*y+25200*x**6*y-27720*x**8*y+11088*x**10*y+1680*y**3-20160*x**2*y**3+75600*x**4*y**3-110880*x**6*y**3+55440*x**8*y**3-10080*y**5+75600*x**2*y**5-166320*x**4*y**5+110880*x**6*y**5+25200*y**7-110880*x**2*y**7+110880*x**4*y**7-27720*y**9+55440*x**2*y**9+11088*y**11]
    if n==49:
             f=lambda x,y:[7*x**6-105*x**4*y**2+105*x**2*y**4-7*y**6,-42*x**5*y+140*x**3*y**3-42*x*y**5]
    fp=lambda x,y: f(x/1000.0,y/1000.0)
    return(fp)

for n in range(17):
    f1=zernike_grad1(n)
    f2=zernike_grad(n)
    x=10.0
    y=50000.0
    print n,numpy.divide(f2(x,y),f1(x,y))

def form_output(mt,dim):
        string=""
        for i in range(dim):
            for j in range(dim):
                if mt[i,j]!=0:
                    string+=str(mt[i,j])
                    if i>0:
                        string+="*x**"+str(i)
                    if j>0:
                        string+="*y**"+str(j)
                    string+="+"
        string=string[:-1]
        if string=="":
            string="0"
        return(string)

def zernike_grad7(degree):
    degree=degree+1
    n=int(ceil(-1.5+0.5*sqrt(1+degree*8)))
    m=n-degree+(n+1)*n/2+1
    l=n-2*m
    #print degree,n,m
    #print "l=",l
    dim=n+1
    fmat=numpy.zeros([dim,dim])
    fmat_x=numpy.zeros([dim,dim])
    fmat_y=numpy.zeros([dim,dim])

    if (l<=0):
        p=0
        q=-l/2 if (n%2==0) else (-l-1)/2
    else:
        p=1
        q=l/2-1 if (n%2==0) else (l-1)/2
    l=abs(l)
    #print "l=",l
    #print "p=",p
    m=(n-l)/2
    #print "mm=",m
    #print "q=",q
    for i in range(q+1):
        for j in range(m+1):
            for k in range(m-j+1):
                #print i,j,k
                factor=1 if ((i+j)%2==0) else -1
                factor=factor*binom(l,2*i+p)
                factor=factor*binom(m-j,k)
                factor=factor*factorial(n-j)/(factorial(j)*factorial(m-j)*factorial(n-m-j))
                y_power=2*(i+k)+p
                x_power=n-2*(i+j+k)-p
                fmat[x_power,y_power]=fmat[x_power,y_power]+factor
                if x_power>0:
                    fmat_x[x_power-1,y_power]=fmat_x[x_power-1,y_power]+factor*x_power
                if y_power>0:
                    fmat_y[x_power,y_power-1]=fmat_x[x_power,y_power-1]+factor*y_power
    #print "i=:",degree,fmat            
    #print form_output(fmat,dim),form_output(fmat_x,dim),form_output(fmat_y,dim)
    
   
   
   
   
   
   
   
    
#for i in range(0,10):
#    f0=zernike_grad(i)
#    f1=zernike_grad1(i)
#    print f0(1000,1000),f1(1000,1000)