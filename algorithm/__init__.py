from . import fcm
from . import pcm
from . import pfcm
from validity import validity
import numpy as np
import pandas as pd

algos={'fcm':fcm, 'pcm':pcm, 'pfcm':pfcm}

def use(name, data, max_iter, num_of_clus):
    if name == 'fcm':
        fcm_object=fcm.fcm(data=data, max_iter=max_iter, num_of_clus=num_of_clus)
        fcm_object.fit()
        return fcm_object
    elif name=='pcm':
        fcm_object=fcm.fcm(data=data, max_iter=max_iter, num_of_clus=num_of_clus)
        fcm_object.fit()
        pcm_object=pcm.pcm(data=data, max_iter=max_iter, num_of_clus=num_of_clus, init_typical=fcm_object.u, init_centroid=fcm_object.v)
        pcm_object.fit()
        return pcm_object
    else:
        fcm_object=fcm.fcm(data=data, max_iter=max_iter, num_of_clus=num_of_clus)
        fcm_object.fit()
        pcm_object=pcm.pcm(data=data, max_iter=max_iter, num_of_clus=num_of_clus, init_typical=fcm_object.u, init_centroid=fcm_object.v)
        pcm_object.fit()
        pfcm_object=pfcm.pfcm(data=data, max_iter=max_iter, num_of_clus=num_of_clus,init_typical=pcm_object.t, init_centroid=fcm_object.v, init_membership=fcm_object.u)
        pfcm_object.fit()

        return pfcm_object


def validity2( data: np.ndarray, membership: np.ndarray, target: np.ndarray):
    tmp=pd.DataFrame(columns=["DB", "PC", "CE", "AC"])
    tmp2=[]
    tmp2.append(validity.davies_bouldin(data, np.argmax(membership, axis=1)))
    tmp2.append(validity.partition_coefficient(membership))
    tmp2.append(validity.classification_entropy(membership))
    tmp2.append(validity.accuracy_score(target, np.argmax(membership, axis=1)))
    tmp.loc[len(tmp)]=tmp2
    tmp.to_csv('ket_qua_phan_cum.csv')
    print(tmp)