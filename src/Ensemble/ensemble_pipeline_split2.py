#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:44:26 2020

@author: denise
"""


'''
This script can be tested locally if the variable cartesius is set to False. If
cartesius is set to true it will work on cartesius.

This script ensembles the hovernet and micronet results (.mat files).
It compares ensemble scores with different relative weights for each model for
the first 100 (or slice_size) images.
and outputs ensemble_score_weight.png in the working directory.

The best weights (that belong to the highest score) are used to ensemble both
models and this is saved as .mat file, it contains a weight for each posible 
class. If you uncomment the line with np.save(....) it saves a .npy file.
ensemble_result.npy (already argmax'ed).

It also outputs images for the plots of each weight devision in .png format.
Only the one which file names end with 0 are used to determine the used 
best_weight. The best weight is based on Jaccard score.
'''
#import sys
#sys.modules[__name__].__dict__.clear()

from scipy.io import loadmat, savemat
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
#import math
import time
import pandas as pd
import sys
 
def initiateVariables(cartesius,split):
    from sys import path
    if cartesius:
        path.append('/home/ccurs011/PanNuke/src/metrics')
        import stats_utils as stats
        data_dir = "/home/ccurs011/HoverNet"
        sub_dir = ["/hover_net/output_split"+str(split)+"/v1.0/np_hv", "/micronet/output_split"+str(split)+"/v3.0/micronet"]
        
        
        if split == 1 or split == 2:
            gt_dir = "/projects/0/ismi2018/PanNuke/Fold-3/masks/"
            gt_sub_dir = ["fold3"] 
            fold = 3
        else:
            gt_dir = "/projects/0/ismi2018/PanNuke/Fold-1/masks/"
            gt_sub_dir = ["fold1"]
            fold = 1
            
        x = ["micronet","np_hv"]
        y = "mask" 
        

        #new:
        slice_size = 100
        begin_slice = 0
        
        nr_images = 2722 #Fold 3 has 2722 images
    else:   
        from sys import path
        path.append('/home/denise/Documents/Vakken/ISMI/PanNuke/src/metrics') #marked
        #from mymodule import myfunction
        # import sys; sys.path.insert(0, '../src/metrics')
        import stats_utils as stats
        data_dir = "/home/denise/Downloads/" #marked
        sub_dir = ["micronet", "np_hv"] #marked
        
        x = ["micronet","np_hv"]
        
        gt_dir = "/home/denise/Downloads/"
        gt_sub_dir = ["output", "PanNuke GT"]
        y = "GT" 
        
        split = 'test'
        
        slice_size = 4
        begin_slice = 0
        nr_images = 12
    return data_dir, sub_dir, gt_dir, gt_sub_dir, x, y, slice_size, begin_slice, nr_images, fold

#%%

def loadValidation(split):
    #load images hover
    data_dir = "/home/ccurs011/HoverNet/"
    sub_directory = ["micronet/output_val"+ str(split),"hover_net/output_val"+str(split)]
    
    #load ground truth
    if split == 1 or split == 3:
        #fold 2
        directory_gt = "/projects/0/ismi2018/PanNuke/Fold-2/masks/fold2"
    elif split == 2:
        #fold 1
        directory_gt = "/projects/0/ismi2018/PanNuke/Fold-1/masks/fold1"
        
    if split == 1:
        slicee = [1392, 1036, 1386, 1170, 1051, 1345, 1363, 1116, 1335, 1332,
                  1202, 1199, 1433, 1448, 1449, 1441, 1280, 1286, 1446, 1273,
                  1464, 1457, 817, 827, 1495, 1486, 1475, 828, 1462, 1500,
                  180, 92, 181, 406, 329, 123, 50, 537, 289, 523, 1174, 1178,
                  1571, 1180, 1193, 1192, 1559, 1563, 1181, 1176, 1810, 1764,
                  739, 702, 741, 2436, 1791, 1672, 1796, 708, 1842, 1105, 1836,
                  1100, 1092, 1085, 1832, 1007, 1825, 875, 1854, 1871, 1885, 1984,
                  1915, 1937, 2013, 1921, 1905, 1877, 2030, 2028, 2039, 2029, 805,
                  2024, 2032, 832, 2015, 2020, 2071, 2082, 2085, 2061, 2064, 2095,
                  2078, 2077, 2107, 2059, 2110, 907, 905, 901, 2113, 916, 921, 897,
                  906, 923, 861, 864, 2141, 2144, 2124, 2121, 863, 2135, 2120, 2129,
                  2212, 2187, 2173, 2191, 2186, 1133, 2155, 2204, 2165, 2182, 2241,
                  2222, 2213, 2220, 2240, 2243, 2227, 2218, 2219, 2214, 2303, 2293,
                  952, 2300, 2260, 2301, 2311, 2309, 2258, 2266, 2320, 2344, 2328,
                  2326, 2333, 2316, 856, 2338, 848, 2337, 1215, 2364, 1217, 1221,
                  1234, 1242, 1232, 1241, 1246, 2379, 2424, 2418, 933, 935, 2412,
                  943, 974, 944, 2399, 984, 930, 931, 932, 2429, 2430, 2431]
    elif split == 2:
        slicee = [1308, 1194, 1209, 1278, 1265, 1574, 1188, 1529, 1175, 1540,
                  1600, 1467, 1619, 1647, 1587, 1392, 1656, 1450, 1379, 1454,
                  1667, 1669, 1662, 932, 1672, 923, 1670, 1673, 1677, 1663,
                  501, 408, 1704, 908, 714, 203, 214, 719, 924, 405, 1776,
                  1352, 1774, 1789, 1773, 1336, 1779, 1783, 1777, 1339, 2631,
                  2651, 844, 1838, 770, 965, 1844, 860, 855, 752, 1134, 2081,
                  1222, 1007, 2086, 2065, 2062, 1017, 1011, 2094, 2117, 2133,
                  2109, 2150, 2112, 2103, 2134, 2160, 2130, 2115, 2173, 2186,
                  2171, 959, 2183, 918, 956, 954, 2188, 955, 2233, 2206, 2245,
                  2197, 2225, 2232, 2240, 2251, 2246, 2218, 892, 880, 891, 893,
                  1050, 878, 1048, 884, 1027, 1025, 977, 2271, 2269, 2311, 2276,
                  2293, 971, 2309, 2270, 2287, 2335, 2359, 2342, 2317, 2324,
                  1294, 2349, 2331, 2353, 1287, 2360, 2386, 947, 2373, 2383,
                  933, 2400, 2395, 941, 2405, 2414, 2437, 2460, 1089, 1090,
                  2445, 2428, 2413, 2462, 2459, 968, 2505, 2496, 2487, 2485,
                  2509, 2470, 2488, 2466, 2506, 1436, 1442, 2520, 1445, 1423,
                  1422, 2534, 1411, 1437, 1425, 1064, 1084, 2563, 1113, 2538,
                  2569, 2548, 2577, 1074, 2551, 2579, 2588, 2585, 2580, 1061,
                  2581, 2582, 2578, 2589, 1062]
    elif split == 3:
        slicee = [1166, 1355, 1116, 1371, 1332, 1111, 1070, 1042, 1394,
                      1336, 1206, 1203, 1292, 1434, 1449, 1430, 1399, 1452,
                      1413, 1280, 1464, 1463, 1491, 1489, 1482, 828, 1495, 
                      1488, 1458, 1461, 174, 100, 247, 467, 223, 1542, 105,
                      207, 278, 113, 1566, 1562, 1190, 1567, 1028, 1023,
                      1559, 1574, 1569, 1172, 1740, 1620, 1634, 1753, 728,
                      1622, 1309, 853, 2511, 2508, 873, 883, 1006, 995, 872,
                      1842, 878, 1005, 1838, 865, 1986, 1949, 1987, 1857, 1888,
                      1956, 1927, 1920, 1925, 1955, 2023, 2031, 833, 803, 806,
                      2035, 805, 2037, 807, 840, 2046, 2077, 2082, 2075, 2055,
                      2064, 2053, 2043, 2095, 2047, 776, 905, 901, 903, 898,
                      912, 2115, 913, 916, 909, 2143, 2128, 2126, 2133, 2134,
                      2141, 2118, 2131, 2136, 2146, 2201, 1147, 2199, 1149,
                      2178, 893, 1124, 1134, 1144, 1143, 2244, 2236, 2237,
                      2227, 2242, 2225, 2221, 2231, 2220, 2233, 954, 957, 2258,
                      2286, 959, 2302, 2250, 2249, 2272, 2264, 2336, 2327,
                      2325, 855, 2333, 2348, 2352, 2320, 859, 2329, 1227, 1244,
                      2362, 1255, 2356, 2379, 2370, 1226, 2371, 1238, 2393,
                      2410, 2428, 2397, 985, 2418, 2409, 942, 987, 2413, 930,
                      931, 932, 2429, 2430, 2431]
    x = ['micronet', 'np_hv']
    y = 'mask'
    validation_GT, validation_hover,validation_micro, temp = getData(data_dir, sub_directory, directory_gt,[''],x,y,slicee,True)
    
    
    return validation_GT, validation_hover, validation_micro
    
def getData(data_dir, sub_dir,gt_dir, gt_sub_dir,x,y,pannuke_slice,cartesius):
    data_dir_list = []
    for folder in sub_dir:
        for root, dirs, files in os.walk(data_dir + folder, topdown=False):
            for name in files:
                #print(os.path.join(root, name))
                if '.mat' in name and '_proc' not in root and '13to100' not in root:
                    data_dir_list.append([int(name.split("_")[1].split(".mat")[0]), os.path.join(root, name)])
    data_dir_list = pd.DataFrame(data_dir_list)
    data_dir_list = data_dir_list.sort_values(by=[0,1])
    data_dir_list1 = data_dir_list.iloc[::2]
    data_dir_list2 = data_dir_list.iloc[1::2]
    data_dir_list1 = data_dir_list1[(data_dir_list1[0]>min(pannuke_slice)) & (data_dir_list1[0]<=max(pannuke_slice)+1)]
    data_dir_list2 = data_dir_list2[(data_dir_list2[0]>min(pannuke_slice)) & (data_dir_list2[0]<=max(pannuke_slice)+1)]
    
    # check for missing files
    missing = []
    if len(list(data_dir_list1[0]))<len(pannuke_slice) or len(list(data_dir_list2[0]))<len(pannuke_slice):
        for i in pannuke_slice:
            if i+1 not in list(data_dir_list1[0]):
                missing.append(i-min(pannuke_slice))
                if i+1 in list(data_dir_list2[0]):
                    data_dir_list2 = data_dir_list2[data_dir_list2[0] != i+1]
            if i+1 not in list(data_dir_list2[0]):
                missing.append(i-min(pannuke_slice))
                if i+1 in list(data_dir_list1[0]):
                    data_dir_list1 = data_dir_list1[data_dir_list1[0] != i+1]
    print('\t missing files from one or both method(s):', [x+min(pannuke_slice)+1 for x in missing])  #[x+1 for x in mylist]                                   
                                                 
        
    output_micro,output_HVnet=[],[]
    del output_micro, output_HVnet
    
    for entry in list(data_dir_list1[1])+list(data_dir_list2[1]):
        if x[0] in entry:
            try:
                output_micro
            except NameError:
                output_micro = loadmat(entry)['result'][:,:,:,:6]
                continue
            output_micro = np.append(output_micro,loadmat(entry)['result'][:,:,:,:6],axis=0)
        if x[1] in entry:
            try:
                output_HVnet
            except NameError:
                output_HVnet = loadmat(entry)['result'][:,:,:,:6]
                continue
            output_HVnet = np.append(output_HVnet, loadmat(entry)['result'][:,:,:,:6],axis=0)    
    
    data_dir_list = []
    for folder in gt_sub_dir:
        for root, dirs, files in os.walk(gt_dir + folder, topdown=False):
            for name in files:
                #print(os.path.join(root, name))
                if '.npy' in name and '13to100' not in root:
                    data_dir_list.append(os.path.join(root, name))
    data_dir_list.sort()     
    
    ground_truth=[]
    del ground_truth    
    
    
    if cartesius:  
        for entry in data_dir_list:
            if y in entry:
                ground_truth = np.load(entry)
                print('2',ground_truth.shape)
                #switch background to front
                ground_truth = ground_truth[:,:,:,[5,0,1,2,3,4]]
        ground_truth = ground_truth.argmax(axis=3)
        print('3',ground_truth.shape)
        ground_truth = ground_truth[pannuke_slice]
        print('4',ground_truth.shape)
        
        
    else:
        for entry in data_dir_list:
            if y in entry:
                try:
                    ground_truth
                except NameError:
                    ground_truth = np.expand_dims(np.load(entry)[:,:,4],axis=2)
                ground_truth = np.append(ground_truth, np.expand_dims(np.load(entry)[:,:,4],axis=2),axis=2)
        ground_truth = ground_truth.T
        ground_truth = ground_truth[pannuke_slice]
    
    ground_truth = np.delete(ground_truth,missing, axis=0) 
    print('Ground truth shape',ground_truth.shape)
    print('Hover shape',output_HVnet.shape)
    print('Micro shape',output_micro.shape)
    
    if len(ground_truth) <= 0:
        try:
            best_weights
        except NameError:
            best_weights = np.array([0,0])
        #continue
    
    return ground_truth, output_HVnet, output_micro, data_dir_list1
  
    
def softvote(models_result, weights=[0.5,0.5]):
    '''
    Parameters
    ----------
    models_result : list of results of the models
    weights : list of weights of each model
        DESCRIPTION. The default is [0.5,0.5], when two models are used,
        otherwise 1/nr_models for each model. (basically hard voting)

    Returns
    -------
    interim : array
        Each row is a pixel, each column shows the number of votes each class
        has got.
    result : array
        Final vote for each pixel

    '''
    interim = np.zeros_like(models_result[0])
    nr_mod = len(models_result)
    if not weights[0]:
        weights = [1/nr_mod]*(nr_mod)
    for model,weight in zip(models_result,weights):
        interim = interim + np.array(model)*weight
    
    result = interim.argmax(axis=3)
    return result, interim

    
def printIntermediateResults(ground_truth, models_label,models):
    jscore = [len(ground_truth)]
    for i, model in enumerate(models_label):
        #score = confusion_matrix(np.reshape(ground_truth, (-1,1)),np.reshape(models[i],(-1,1)),normalize='all')
        #score = (stats.get_fast_dice_2(np.reshape(ground_truth, (-1,1)),np.reshape(models[i],(-1,1)))+
        #         stats.get_fast_aji(np.reshape(ground_truth, (-1,1)),np.reshape(models[i],(-1,1)))+
        #         stats.get_fast_aji_plus(np.reshape(ground_truth, (-1,1)),np.reshape(models[i],(-1,1)))) #marked
        jscore.append(jaccard_score(np.reshape(ground_truth, (-1,1)),np.reshape(models[i],(-1,1)),average='micro'))
        print('\t Jaccard score ' + model + ': ', jscore)
    # plt.imshow(score)
    # plt.show()
    # plt.imshow(score[1:,1:])
    # plt.show()
    return jscore

    
def trainSoftvote(ground_truth, models_result, weights=[None,None],score=[],nr_steps=10):
    '''only works for 2 models...'''
    nr_mod = len(models_result)
    if not weights[0]:
        weights = [0]*(nr_mod)
        weights[0] = 1
        weights = [weights]
    ensemble = softvote(models_result, weights = weights[-1])[0]
    #print(np.unique(ensemble), np.unique(ground_truth))
    score.append(jaccard_score(ground_truth.reshape((-1,1)),ensemble.reshape((-1,1)),average='micro')) #marked
    new_weights = [weights[-1][0]-(1/nr_steps), weights[-1][1]+(1/nr_steps)]
    new_weights = [round(num,int(np.log10(nr_steps))) for num in new_weights]
    #print(new_weights,weights)
    weights.append(new_weights)
    if new_weights[0]<0:
        del weights[-1]
        return np.array(weights),score
    else:
        return trainSoftvote(ground_truth,models_result,weights,score,nr_steps=nr_steps)

    
def weightsVSscores(weights,score,subset):
    fig = plt.figure()
    
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    
    ax1.plot(weights[:,0],score[-len(weights[:,0]):])
    ax1.set_xlabel(r"Weight of Hover-Net")
    ax1.set_xlim(0.0,1.0)
    
    ax2.set_xlim(1.0,0.0)
    ax2.set_xlabel(r"Weight of Micro-Net") #marked
    ax2.plot(weights[:,1],score[-len(weights[:,1]):], '.')
    
    ax1.set_ylabel('Ensemble (Jaccard) Score') #marked
    plt.savefig('./ensemble_score_weight'+str(subset)+'.png',dpi=200,bbox_inches='tight')
    plt.show()
    return

    
def saveResults(data_dir_list1,ensemble_interim,ensemble_output,split,fold,soft_vote_score,hard_vote_score):
    #np.save('ensemble_result.npy',ensemble_trained) #marked
    soft_vote_score = pd.DataFrame(soft_vote_score,columns=['nr_images', 'micro','hover','ensemble'])
    hard_vote_score = pd.DataFrame(hard_vote_score,columns=['nr_images', 'micro','hover','ensemble'])
    
    soft_vote_score.to_csv('./EnsembleResults/split'+str(split)+'/fold' + str(fold)+'_softvote.csv')
    hard_vote_score.to_csv('./EnsembleResults/split'+str(split)+'/fold' + str(fold)+'_hardvote.csv')
    
    
    # for image_nr,image in zip(list(data_dir_list1[0]),ensemble_interim):
    #     savemat('./EnsembleResults/split'+str(split)+'/fold' + str(fold)+'_'+str(image_nr)+'.mat', {"result": image})
        #np.save('./EnsembleResults/split'+str(split)+'/fold' + str(fold)+'_'+str(image_nr)+'.npy', ensemble_output.astype(np.uint8))
    return
    
def classes2instances(ensemble_trained): #not perfect
    count = 1
    output = np.zeros([ensemble_trained.shape[0],ensemble_trained.shape[1],ensemble_trained.shape[2],2]).astype(int)
    for i,image in enumerate(ensemble_trained):
        output[i,:,:,0] = np.where(image == 0, 0, np.nan)
        for pixeli in range(ensemble_trained.shape[1]):
            for pixelj in range(ensemble_trained.shape[2]):
                if np.isnan(output[i,pixelj,pixeli,0]):
                    floodfill(output[i,:,:,0], pixelj, pixeli,count)
                    count+=1
    output[:,:,:,1] = ensemble_trained
    return output.astype(np.uint16)


def floodfill(matrix, x, y,count): #source https://stackoverflow.com/questions/19839947/flood-fill-in-python
    if np.isnan(matrix[x][y]):  
        matrix[x][y] = count 
        #recursively invoke flood fill on all surrounding pixels:
        if x > 0:
            floodfill(matrix,x-1,y,count)
        if x < len(matrix[y]) - 1:
            floodfill(matrix,x+1,y,count)
        if y > 0:
            floodfill(matrix,x,y-1,count)
        if y < len(matrix) - 1:
            floodfill(matrix,x,y+1,count)
        
    
#%% main
def main():
    tic = time.time()
    cartesius = True
    hard_vote_score, soft_vote_score = [[0,0,0,0]],[[0,0,0,0]]
    print('cartesius is', cartesius, '\n')
    print('old recursion limit:',sys.getrecursionlimit())
    sys.setrecursionlimit(256*256)
    print('new recursion limit:',sys.getrecursionlimit())
    split = 2
    data_dir, sub_dir, gt_dir, gt_sub_dir, x, y, slice_size, begin_slice, nr_images, fold = initiateVariables(cartesius,split)
    
    validation_GT, validation_hover,validation_micro = loadValidation(split)
    print(f'After {time.time()-tic:.2f} seconds start train softvote')
    weights, score = trainSoftvote(validation_GT,[validation_hover,validation_micro])
    print(f'After {time.time()-tic:.2f} seconds start plot scores per weight')
    weightsVSscores(weights, score, split)
    #loop over subsets to avoid to much data loaded at the same time.
    for subset in range(nr_images//slice_size+1):
        pannuke_slice = range(subset*(slice_size),min((1+subset)*slice_size,nr_images))
        print(pannuke_slice)
        try:
            min(pannuke_slice)
        except ValueError:
            break

        print(f'After {time.time()-tic:.2f} seconds start get data')
        ground_truth, output_HVnet, output_micro, data_dir_list1 = getData(data_dir, sub_dir,gt_dir, gt_sub_dir,x,y,pannuke_slice,cartesius)
        
        print(f'After {time.time()-tic:.2f} seconds start do softvote 50/50')
        ensemble = softvote([output_HVnet, output_micro])[0]
        
        print(f'After {time.time()-tic:.2f} seconds start Scores of all methods')
        hard_vote_score.append(printIntermediateResults(ground_truth,['Micro','Hover', 'Ensemble'],[output_micro.argmax(axis=3), output_HVnet.argmax(axis=3), ensemble]))
        
        
        print(f'After {time.time()-tic:.2f} seconds start use best weight')
        if subset==0:
            best_weights = weights[score.index(max(score))]
            print('\t the best weights are: ', best_weights)
        ensemble_trained, ensemble_interim = softvote([output_HVnet, output_micro], best_weights)
        
        
        if subset==0:
            plt.subplot(1,5,1)
            plt.imshow(ground_truth[0])
            plt.axis('off')
            plt.title('Ground Truth')
            plt.subplot(1,5,2)
            plt.imshow(output_HVnet[0].argmax(axis=2))
            plt.axis('off')
            plt.title('HoverNet')
            plt.subplot(1,5,3)
            plt.imshow(output_micro[0].argmax(axis=2)) 
            plt.axis('off')
            plt.title('MicroNet')
            plt.subplot(1,5,4)
            plt.imshow(ensemble[0]) #hardvote
            plt.axis('off')
            plt.title('Hardvote')
            plt.subplot(1,5,5)
            plt.imshow(ensemble_trained[0])
            plt.axis('off')
            plt.title('Softvote')
            plt.savefig('./exampleresult'+str(split)+'.png',dpi=200,bbox_inches='tight')
            plt.show()
        
        #new def ensemble classes to instances 
        print(f'After {time.time()-tic:.2f} seconds start convert classes to instances')
        ensemble_output = classes2instances(ensemble_trained)        
        
        
        print(f'After {time.time()-tic:.2f} seconds start Scores of all methods Ensemble with best weights')
        soft_vote_score.append(printIntermediateResults(ground_truth, ['Micro','Hover', 'Ensemble'],[output_micro.argmax(axis=3), output_HVnet.argmax(axis=3), ensemble_trained]))
        
        print(f'After {time.time()-tic:.2f} seconds start save results')
        saveResults(data_dir_list1,ensemble_interim,ensemble_output,split,fold,soft_vote_score,hard_vote_score)
        
        
        image_nr = max(list(data_dir_list1[0]))
        toc = time.time()
        total_time = toc-tic
        total_time_est = (total_time)/(image_nr)*nr_images
        delta_time = total_time_est - total_time
        print(f'Run-time is {total_time:.2f} seconds for {image_nr} PanNuke images.' +
          f' ETA = {delta_time:.2f} seconds \n \n')

#%%
if __name__ == "__main__":
    # execute only if run as a script
    main()