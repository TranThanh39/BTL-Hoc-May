import numpy as np
import pandas as pd
from validity import validity
from data_process import image_process, show_data
from .fcm import fcm

from .pcm import pcm


def normalize_data(data, d):
    min_v=np.min(data)
    max_v=np.max(data)
    return d*(data-min_v)/(max_v-min_v)


class pfcm(pcm):
    def __init__(self, data, max_iter, num_of_clus, init_typical, init_centroid, init_membership, eps=1e-4, m=2, a=1, b=1, n=2):
        super().__init__(data=data, max_iter=max_iter, num_of_clus=num_of_clus, eps=eps, m=m, init_typical=init_typical, init_centroid=init_centroid)
        self.u=init_membership
        self.a, self.b, self.n = a, b, n
        self.v=init_centroid


    def update_v(self):
        u_pw_m= self.a*(self.u.T ** self.m) + self.b*(self.t.T ** self.n)
        tu=np.dot(u_pw_m, self.data)
        mau=np.sum(u_pw_m, axis=1, keepdims=True)
        return tu/mau
    
    def update_t(self):
        return 1/(1+(self.b*(self.distance**2)/self.gamma)**(1/(self.n-1)))

    def update_u(self):
        tu=np.linalg.norm(self.data[:, np.newaxis, :]-self.v, axis=2)**2
        mau=tu.T[:, : , np.newaxis]
        return 1/np.sum((tu/mau) ** (2/(self.m-1)), axis=0)

    def fit(self):
        self.normalize_distance(d=1)
        self.gamma=self.caculate_gamma()
        for i in range(self.max_iter):
            self.normalize_distance(d=2)
            old_v=self.v.copy()
            self.u=self.update_u()
            # self.t=normalize_data(self.update_t(), d=4)
            self.t=self.update_t()
            self.v=self.update_v()
            tmp2=np.linalg.norm(self.v-old_v)
            print(i, tmp2)

            if tmp2<self.eps:
                return i
        return i


if __name__== '__main__':
    # np.random.seed(42)
    # data_origin=pd.read_csv('data/data2.csv')
    # data, target=np.array(data_origin.iloc[:, 0:2]), pd.factorize(np.array(data_origin.iloc[:,2]))[0]

    # fcm_object=fcm(data=data, max_iter=1000, num_of_clus=2, eps=1e-3, m=2)
    # fcm_object.fit()
    # pcm_object=pcm(data=data, max_iter=1000, num_of_clus=2, eps=1e-3, m=2, init_typical=fcm_object.u, init_centroid=fcm_object.v)
    # pcm_object.fit()
    # pfcm_object=pfcm(data=data, max_iter=1000, num_of_clus=2, eps=1e-3, m=2, init_typical=pcm_object.t, init_centroid=pcm_object.v, init_membership=fcm_object.u, a=1, b=1, n=2)
    # pfcm_object.fit()

    # data_origin['z']=np.argmax(pfcm_object.u, axis=1)
    # show_data.scatter_chart(data=data_origin, centroids=pfcm_object.v, fig_name='nhap_loz.png' )

   
    np.random.seed(42)
    imgpr=image_process.image_pr(['data/b1_1024x1024.tif', 'data/b2_1024x1024.tif', 'data/b3_1024x1024.tif', 'data/b4_1024x1024.tif'])
    color=np.array([[0, 128, 0, 255],[128, 128, 128, 255],[0, 255, 0, 255],[1, 192, 255, 255],[0, 0, 255, 255],[0, 64, 0, 255]])
    data=imgpr.read_image(mode=1)

    fcm_object=fcm(data=data, max_iter=100, num_of_clus=6, eps=1e-3, m=2)
    fcm_object.fit()
    pcm_object=pcm(data=data, max_iter=100, num_of_clus=6, eps=1e-3, m=2, init_typical=fcm_object.u, init_centroid=fcm_object.v)
    pcm_object.fit()
    pfcm_object=pfcm(data=data, max_iter=1000, num_of_clus=6, eps=1e-3, m=2, init_typical=fcm_object.u, init_centroid=fcm_object.v, init_membership=fcm_object.u, a=1, b=1, n=2)
    pfcm_object.fit()

    imgpr.process(list_u=[pfcm_object.u], list_v=[pfcm_object.v], num_of_data_site=1, name_output='result/anhvientham.png', color=color)
    





