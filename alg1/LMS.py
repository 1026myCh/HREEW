import math
import numpy as np
class LMS():
    def __init__(self):
        self.w1=0
        self.b1=1
        self.mu=0.8
        self.alpha=0.97
        self.gama=0.8
        self.beta=0.2
        self.mu_min=0.01
        self.mu_max=0.3
        self.p=0
        self.err=0
    def LMS_Filter(self,data_in):
        data=np.copy(data_in)
        if data>1.0:
            data=1.0
        y=self.w1*self.b1
        e=(data-y)
        tempw1=self.w1+self.mu*e*self.b1#tempw1= LMS.w1+LMS.mu*e*LMS.b1;
        tempmu=self.mu*self.alpha+self.gama*self.p#LMS.mu*LMS.alpha+LMS.gama*LMS.p;
        tmpp=self.beta*self.p+(1-self.beta)*e*self.err#LMS.beta*LMS.p+(1-LMS.beta)*e*LMS.err
        if math.isnan(tempw1)==False:
            self.w1=np.copy(tempw1)
        if math.isnan(tmpp)==False:
            self.p=np.copy(tmpp)
        if math.isnan(tempmu)==False:
            self.mu=np.copy(tempmu)
        if self.mu>self.mu_max:
            self.mu=np.copy(self.mu_max)
        if self.mu<self.mu_min:
            self.mu=np.copy(self.mu_min)
        self.err=e
        return e




