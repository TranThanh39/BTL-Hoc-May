from algorithm import *
import numpy as np
from data_process import image_process, show_data
import pandas as pd
from sklearn.datasets import load_iris

def anh_vien_tham(max_iter, num_of_clus, algos_name, result_name):
    np.random.seed(42)
    imgpr=image_process.image_pr(['data/b1_1024x1024.tif', 'data/b2_1024x1024.tif', 'data/b3_1024x1024.tif', 'data/b4_1024x1024.tif'])
    color=np.array([[0, 128, 0, 255],[128, 128, 128, 255],[0, 255, 0, 255],[1, 192, 255, 255],[0, 0, 255, 255],[0, 64, 0, 255]])
    data=imgpr.read_image(mode=1)
    algo=use(algos_name, data=data, max_iter=max_iter, num_of_clus=num_of_clus)
    imgpr.process(list_u=[algo.u], list_v=[algo.v], num_of_data_site=1, name_output='result/'+result_name+'jpg', color=color)
    


def data_binh_thuong(max_iter, num_of_clus, algos_name, result_name):
    np.random.seed(42)
    data_origin=pd.read_csv('data/data2.csv')
    data, target=np.array(data_origin.iloc[:, 0:2]), pd.factorize(np.array(data_origin.iloc[:,2]))[0]
    algo=use(algos_name, data=data, max_iter=max_iter, num_of_clus=num_of_clus)
    if algos_name == 'pcm':
        data_origin['z']=np.argmax(algo.t, axis=1)
    else:
        data_origin['z']=np.argmax(algo.u, axis=1)
    show_data.scatter_chart(data=data_origin, centroids=algo.v, fig_name='result/'+result_name+'jpg' )


def data_iris(max_iter, num_of_clus, algos_name, result_name):
    np.random.seed(42)
    tmp=load_iris()
    data, target=tmp['data'], tmp['target']
    algo=use(algos_name, data=data, max_iter=max_iter, num_of_clus=num_of_clus)
    if algos_name == 'pcm':
        validity2(data=data, membership=algo.t, target=target, result_name=result_name)
    else:
        validity2(data=data, membership=algo.u, target=target, result_name=result_name)




if __name__=='__main__':

    #Dùng 1 trong 3 dòng code để chạy thư nghiệm
        #Dòng 1 phân cụm cho bộ dữ liệu iris
        #Dòng 2 phân cụm cho bộ dữ liệu tự tạo
        #Dòng 3 phân cụm cho dữ liệu ảnh viễn thám

    #Dữ liệu nằm trong folder data
    
    #Giải thích tham số
        #max_iter là sô lần lặp tối đa
        #num_of_clus là số cụm mong muốn
        #algos_name là thuật toán muốn sử dụng (một trong 3 cái fcm, pcm, pfcm)
        #result_name là tên của ảnh kêt quả đầu ra (kết quả đầu ra nằm trong folder result)



    # data_iris(max_iter=1000, num_of_clus=3, algos_name='pcm', result_name='a')
    # data_binh_thuong(max_iter=1000, num_of_clus=2, algos_name='fcm', result_name='a')
    anh_vien_tham(max_iter=1000, num_of_clus=6, algos_name='fcm', result_name='anh_vien_tham_cac_kieu')
