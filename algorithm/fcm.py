import numpy as np
import pandas as pd
from data_process import image_process, show_data
np.random.rand(42)


class fcm():
    def __init__(self, data, max_iter, num_of_clus, eps=1e-4, m=2, init_centroid=None):
        self.max_iter=max_iter
        self.eps=eps
        self.m=m
        self.data=data
        self.num_of_clus=num_of_clus
        self.init_centroid=init_centroid
        

    def initial(self):
        self.u=np.random.rand(len(self.data), self.num_of_clus)
        self.u=self.u/np.sum(self.u, axis=1, keepdims=True)
        if self.init_centroid is not None:
            self.v=self.init_centroid
        else:
            self.v=self.update_v()


    def update_v(self):

        u_pw_m=self.u.T ** self.m
        tu=np.dot(u_pw_m, self.data)
        mau=np.sum(u_pw_m, axis=1, keepdims=True)
        return tu/mau


    def update_u(self):
        tu=np.linalg.norm(self.data[:, np.newaxis, :]-self.v, axis=2)
        mau=tu.T[:, : , np.newaxis]
        return 1/np.sum((tu/mau) ** (2/(self.m-1)), axis=0)



    def fit(self):
        self.initial()
        for i in range(self.max_iter):

            old_u=self.u.copy()
            self.u=self.update_u()
            self.v=self.update_v()

            tmp2=np.linalg.norm(self.u-old_u, ord=2)
            if tmp2<self.eps:
                return self.u, self.v, i
        return self.u, self.v, i
    

if __name__=='__main__':
    np.random.seed(42)
    data_origin=pd.read_csv('data/data2.csv')
    data, target=np.array(data_origin.iloc[:, 0:2]), pd.factorize(np.array(data_origin.iloc[:,2]))[0]
    fcm_object=fcm(data=data, max_iter=1000, num_of_clus=2, eps=1e-3, m=2)
    fcm_object.fit()

    data_origin['z']=np.argmax(fcm_object.u, axis=1)
    
    show_data.scatter_chart(data=data_origin, centroids=fcm_object.v.T, fig_name='nhapfcm.png' )

    
    