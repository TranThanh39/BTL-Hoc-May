import numpy as np
import pandas as pd
import math
from .pcm import pcm
from .fcm import fcm

import validity
from data_process import show_data


def min_max_normalize(data, value):
    min_v, max_v = np.min(data), np.max(data)
    return 0.1+((data-min_v)/(max_v-min_v))*value


class gspfcm():
    
    def __init__(self, data: np.ndarray,label: np.ndarray, num_of_clus: int, 
                 ratio_label:float, a: float =1 , b: float = 1, eta: float = 2, K:float = 1,
                 m: float = 2, eps: float = 1e-5, gamma=1, max_iter: int = 10000):
        
        self.data, self.label, self.num_of_clus, self.ratio_label = data, label, num_of_clus, ratio_label
        self.a, self.b, self.eta, self.gamma= a, b, eta, gamma
        self.K, self.m, self.eps, self.maxiter = K, m, eps, max_iter

        self.fcm_object=fcm(data=data, max_iter=1000, num_of_clus=self.num_of_clus, eps=1e-3, m=2)
        self.fcm_object.fit()

        self.run_pcm=pcm(data=self.data, max_iter=1000, num_of_clus=self.num_of_clus, eps=1e-3, m=2, init_typical=self.fcm_object.u, init_centroid=self.fcm_object.v)


    def init_semi_data(self):

        list_label = np.unique(self.label)
        index = [np.where(self.label==i)[0] for i in list_label]

        while True:
            try:
                x=np.random.rand(self.num_of_clus)
                x=np.ceil((x/np.sum(x))*(self.ratio_label/100)*len(self.data))

                self.random_index=[np.random.choice(len(i), size=int(j), replace=False) for i, j in zip(index, x)]

                self.semi_data=[self.data[index[i][j]] for i,j in zip(range(self.num_of_clus), self.random_index)]
            except ValueError:
                continue
            break


        self.v_semi=np.array([np.mean(i, axis=0) for i in self.semi_data])
        
        

    def caculate_u_semi(self, distance: np.ndarray  ):
        distance=np.power(distance, 2/(self.m-1))
        u_semi=1/np.sum(distance[:, :, np.newaxis]/distance[:, np.newaxis, :], axis=2)
        tmp=np.argmax(u_semi, axis=1)
        self.u_semi=np.zeros_like(u_semi)
        for i, j in zip(range(len(u_semi)), tmp):
            self.u_semi[i, j]=u_semi[i, j]


    def caculate_t(self, distance: np.ndarray, u:np.ndarray):
        distance_cp=min_max_normalize(distance, value=2)
        ray=self.K * np.sum(u ** self.eta * distance_cp ** 2, axis=0) / np.sum(u ** self.eta, axis=0)
        distance_cp=2*distance_cp
        return 1/(1+((self.b * distance_cp ** 2 )/ ray) ** (1/(self.eta-1)))
        

    def update_u(self, distance: np.ndarray):
        component_1=(1/distance**2)**(1/(self.m-1))
        # tmp=1-np.sum(self.u_star, axis=1, keepdims=True)
        # tmp[np.where(np.abs(tmp-1)<1e-5)[0].tolist()]=1
        return self.u_semi + (1-np.sum(self.u_semi, axis=1, keepdims=True))*component_1/np.sum(component_1, axis=1, keepdims=True)


    def update_t(self):
        distance=np.linalg.norm(self.data[:, np.newaxis, :]-self.v, axis=2)+np.linalg.norm(self.v-self.v_semi, axis=1)*self.gamma
        common=(self.nuy/(self.b*distance**2))**(1/(self.eta-1))
        tmp=np.where(self.t<self.t_semi)
        common[tmp]=-common[tmp]
        return (self.t_semi+common)/(1+common)
    

    def update_v(self):
        common: np.ndarray = self.a*np.abs(self.u-self.u_semi)**self.m + self.b*np.abs(self.t-self.t_semi)**self.eta
        tu=np.sum((self.data + self.v_semi[:, np.newaxis, :])*common.T[:, :, np.newaxis], axis=1)
        mau=np.sum(common.T*(self.gamma+1), axis=1, keepdims=True)
        return tu/mau


    def check_iter(self, u_old, t_old):
        tmp=np.linalg.norm(self.u-u_old) + np.linalg.norm(self.t-t_old)
        if tmp<self.eps:
            return True
        return False



    def fit(self):
        self.init_semi_data()

        distance=np.linalg.norm(data[:, np.newaxis, :]-self.v_semi, axis=2)
        distance=min_max_normalize(data=distance, value=2)

        self.caculate_u_semi(distance=distance)
        self.t_semi=self.caculate_t(distance=distance, u=self.u_semi)


        self.run_pcm.fit()
        self.u, self.v = self.fcm_object.u, self.run_pcm.v

        
        distance=np.linalg.norm(data[:, np.newaxis, :]-self.v, axis=2)
        distance=min_max_normalize(distance, value=1)

        self.nuy=self.K * np.sum(self.u ** self.eta * distance ** 2, axis=0) / np.sum(self.u ** self.eta, axis=0)

        self.t=self.caculate_t(distance=distance, u=self.u)

        for i in range(self.maxiter):
            print(i)
            distance=np.linalg.norm(data[:, np.newaxis, :]-self.v, axis=2)
            distance=min_max_normalize(distance, value=2)
            u_old=self.u.copy()
            t_old=self.t.copy()
            self.v=self.update_v()
            self.u=self.update_u(distance=distance)
            self.t=self.update_t()
            if self.check_iter(u_old=u_old, t_old=t_old):
                return i
        return i
    
        




if __name__ == "__main__":


    np.random.rand(42)
    data_origin=pd.read_csv('data/data3.csv')
    # data_origin2=pd.read_csv('data/data3.csv')

    data, target=np.array(data_origin.iloc[:, 0:2]), pd.factorize(np.array(data_origin.iloc[:,2]))[0]
    # data2, target2=np.array(data_origin2.iloc[:, 0:2]), pd.factorize(np.array(data_origin2.iloc[:,2]))[0]

    gspfcm_object=gspfcm(data=data, label=target, ratio_label=50, num_of_clus=2, max_iter=1000, b=1)
    gspfcm_object.fit()
    
    print(gspfcm_object.u)

    print(gspfcm_object.t)


    data_origin['z']=np.argmax(gspfcm_object.u, axis=1)
    
    show_data.scatter_chart(data=data_origin, centroids=gspfcm_object.run_pcm.v, fig_name='nhap.png' )
    print(gspfcm_object.run_pcm.v)









    
        