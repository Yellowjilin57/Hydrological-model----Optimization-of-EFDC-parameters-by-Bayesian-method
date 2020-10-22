# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 16:13:48 2020

@author: JLIN
"""

import sys
sys.path.append(r"D:\Spyder")
import os
import numpy as np
import pandas as pd
import math
import sys
from bayes_opt1 import BayesianOptimization
from bayes_opt1.util import UtilityFunction, Colours
from bayes_opt1 import BayesianOptimization
from bayes_opt1.util import load_logs
from bayes_opt1.logger import JSONLogger
from bayes_opt1.event import Events
import  matplotlib.pyplot as plt
import  matplotlib
import time  # 引入time模块


##先给变量赋予其本身的初始值，通过敏感性分析筛选出19个指标
X01=0;X02=0;X03=0;X04=0;X05=0;X06=0;X07=0;X08=0;X09=0;X10=0;X11=0;X12=0;X13=0;X14=0;X15=0;X16=0;X17=0;X18=0;X19=0;X20=0;

##这是保存输出过程为外部文件的主程序
'''
class Logger(object):
    def __init__(self, fileN="HJL_finalresult.txt"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
'''

def black_box_function(X01,X02,X03,X04,X05,X06,X07,X08,X09,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20):
    
    fileWENJIANpath = r'C:\Users\PKUWSL-JLin\Desktop\AUTO\MODEL\K1_1'
    shuizhi_Dataframe = r'C:\Users\PKUWSL-JLin\Desktop\AUTO\MODEL'
    picturepath = r'C:\Users\PKUWSL-JLin\Desktop\AUTO\Figure\F1'
    withopen_nash = r'C:\Users\PKUWSL-JLin\Desktop\AUTO\RESULT\NASH_Out1.txt'
    withopen_seven = r'C:\Users\PKUWSL-JLin\Desktop\AUTO\RESULT\Seven_1.txt.txt'
    Iwindpath = fileWENJIANpath
    path = fileWENJIANpath
    #JSONpath = 
    
    
    ########################################+========================================================================================
    ########################################+========================================================================================
    ########################################+========================================================================================
    ########################################+========================================================================================
    ########################################+========================================================================================
    
    dict1 = {'first part': "#****************集成的流域智能设计（IWIND）系统--湖库/河流富营养化模型输入模板  ***********\n"
                 "20	1	0	-1"
                           "\n1  1  1  1	1  1  1	1	1	1	1	1	1	1	1	1	1	1	0	0	1	1	0	1	0	0 \n"
                           "1	1	0	2	2	0	1	0	\n"
                           "1	0	1	813	0	0.7	1	0.1	0	\n"
                           "1	3	\n"
                           "20	0	10000	24	1\n"
                           "6	17 \n"
                           "14   19 \n"
                           "20	11\n"
                           "10	16\n"
                           "17	15\n"
                           "23	16\n"
                           "31	17\n"
                           "36	20\n"
                           "43	21\n"
                           "50	17\n"
                           "56	16\n"
                           "36	27\n"
                           "30	11\n"
                           "25	5\n"
                           "29	4\n"
                           "12	22\n"
                           "27	25\n"
                           "34	10\n"
                           "41	26\n"
                           "52	11\n"}
    dict2 = {'third part': '0	0	0	0	0	0	\n'
                           '8	8\n'
                           '8	22	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0\n'
                           '10	21	0	2	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0\n'
                           '7	22	0	3	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0\n'
                           '14	21	0	4	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0\n'
                           '39	28	0	5	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0\n'
                           '40	12	0	6	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0\n'
                           '11	23	0	7	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0\n'
                           '41	13	0	8	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0\n'
                           '0	0  0  0  0	0  0  0	0	0	0	0	0.00019	0	0	0	0	0	0.0069	0	0	0	0	0	0	0	0\n'
                           ' 0	0	0	0	0	0	0	0	0	0	0	0.039	0	0	0	0	0	0.95	0	0	0	0	0	0	0	0\n'
                           'WQWCRST.out	模型输出的水质重启文件\n'
                           'NONE	初始条件设置文件\n'
                           'NONE	藻类生长，呼吸，捕食文件\n'
                           'NONE	藻类与有机物沉降速率文件\n'
                           'NONE	太阳辐射相关输入文件\n'
                           'NONE	底泥通量输入文件\n'
                           'WQPSC.inp	点源边界条件文件\n'
                           'NONE	非点源（包括大气沉降文件\n'
                           'NONE	色度文件\n'
                           'NONE	负数诊断文件\n'
                           ' 0.0005	0.0005	0.0005	0.0007	0.0009	0.0007	0	0	0	350	500	20	0.07	0.1	0.1	2	2	0.5 '}
    
    ##### 生成文件一：水质模型板块的原始文件
    
    df = pd.read_excel(shuizhi_Dataframe + '\shuizhi.xlsx',sep='\s+',header=None)
    dft = pd.DataFrame(df)
    c08_j = dft.iloc[0,1:11]
    c08_z = dft.iloc[1,1:11]
    c09_j = dft.iloc[2,1:13]
    c09_z = dft.iloc[3,1:13]
    c10_j = dft.iloc[4,1:10]
    c10_z = dft.iloc[5,1:10]
    c11_j = dft.iloc[6,1:11]
    c11_z = dft.iloc[7,1:11]
    c12_j = dft.iloc[8,1:11]
    c12_z = dft.iloc[9,1:11]
    c13_j = dft.iloc[10,1:9]
    c13_z = dft.iloc[11,1:9]
    c14_j = dft.iloc[12,1:10]
    c14_z = dft.iloc[13,1:10]
    c15_j = dft.iloc[14,1:6]
    c15_z = dft.iloc[15,1:6]
    c16_j = dft.iloc[16,1:8]
    c16_z = dft.iloc[17,1:8]
    c17_j = dft.iloc[18,1:8]
    c17_z = dft.iloc[19,1:8]
    c18_j = dft.iloc[20,1:11]
    c18_z = dft.iloc[21,1:11]
    c19_j = dft.iloc[22,1:7]
    c19_z = dft.iloc[23,1:7]
    c20_j = dft.iloc[24,1:11]
    c20_z = dft.iloc[25,1:11]
    c21_j = dft.iloc[26,1:14]
    c21_z = dft.iloc[27,1:14]
    c22_j = dft.iloc[28,1:11]
    c22_z = dft.iloc[29,1:11]
    c23_j = dft.iloc[30,1:7]
    c23_z = dft.iloc[31,1:7]
    c24_j = dft.iloc[32,1:13]
    c24_z = dft.iloc[33,1:13]
    c25_j = dft.iloc[34,1:8]
    c25_z = dft.iloc[35,1:8]
    c26_j = dft.iloc[36,1:7]
    c26_z = dft.iloc[37,1:7]
    c27_j = dft.iloc[38,1:10]
    c27_z = dft.iloc[39,1:10]
    c28_j = dft.iloc[40,1:11]
    c28_z = dft.iloc[41,1:11]
    c29_j = dft.iloc[42,1:9]
    c29_z = dft.iloc[43,1:9]
    c30_j = dft.iloc[44,1:30]
    c30_z = dft.iloc[45,1:30]
    c31_j = dft.iloc[46,1:21]
    c31_z = dft.iloc[47,1:21]
    c32_j = dft.iloc[48,1:9]
    c32_z = dft.iloc[49,1:9]
    C08 = dict(zip(c08_j,c08_z))
    C09 = dict(zip(c09_j,c09_z))
    C10 = dict(zip(c10_j,c10_z))
    C11 = dict(zip(c11_j,c11_z))
    C12 = dict(zip(c12_j,c12_z))
    C13 = dict(zip(c13_j,c13_z))
    C14 = dict(zip(c14_j,c14_z))
    C15 = dict(zip(c15_j,c15_z))
    C16 = dict(zip(c16_j,c16_z))
    C17 = dict(zip(c17_j,c17_z))
    C18 = dict(zip(c18_j,c18_z))
    C19 = dict(zip(c19_j,c19_z))
    C20 = dict(zip(c20_j,c20_z))
    C21 = dict(zip(c21_j,c21_z))
    C22 = dict(zip(c22_j,c22_z))
    C23 = dict(zip(c23_j,c23_z))
    C24 = dict(zip(c24_j,c24_z))
    C25 = dict(zip(c25_j,c25_z))
    C26 = dict(zip(c26_j,c26_z))
    C27 = dict(zip(c27_j,c27_z))
    C28 = dict(zip(c28_j,c28_z))
    C29 = dict(zip(c29_j,c29_z))
    C30 = dict(zip(c30_j,c30_z))
    C31 = dict(zip(c31_j,c31_z))
    C32 = dict(zip(c32_j,c32_z))

    ##### 生成文件二：水生植物板块的原始文件
    da = pd.read_excel(shuizhi_Dataframe + '\zhiwu.xlsx',sep='\s+')
    dag = pd.DataFrame(da)
    z01_j = dag.iloc[0,0:3]
    z01_z = dag.iloc[1,0:3]
    z02_j = dag.iloc[2,0:9]
    z02_z = dag.iloc[3,0:9]
    z03_j = dag.iloc[4,0:6]
    z03_z = dag.iloc[5,0:6]
    z04_j = dag.iloc[6,0:4]
    z04_z = dag.iloc[7,0:4]
    z05_j = dag.iloc[8,0:10]
    z05_z = dag.iloc[9,0:10]
    z06_j = dag.iloc[10,0:9]
    z06_z = dag.iloc[11,0:9]
    z07_j = dag.iloc[12,0:4]
    z07_z = dag.iloc[13,0:4]
    z08_j = dag.iloc[14,0:19]
    z08_z = dag.iloc[15,0:19]
    Z01 = dict(zip(z01_j,z01_z))
    Z02 = dict(zip(z02_j,z02_z))
    Z03 = dict(zip(z03_j,z03_z))
    Z04 = dict(zip(z04_j,z04_z))
    Z05 = dict(zip(z05_j,z05_z))
    Z06 = dict(zip(z06_j,z06_z))
    Z07 = dict(zip(z07_j,z07_z))
    Z08 = dict(zip(z08_j,z08_z))
    
        ###开始变量的调参，并将变量值写入字典，然后写入外部文件，然后执行文件 
    C17['KTHDR'] = X01
    C17['KTMNL'] = X02
    C17['KHDNN'] = X03
    C25['ANDC'] = X04
    C25['rNitM'] = X05
    C25['KHNitN'] = X06
    C25['KNit2'] = X07
    C26['KDN'] = X08
    C32['WSrp'] = X09
    C32['WSlp'] = X10
    C32['RNPREF'] = X11
    C16['KDC'] = X12
    C26['KRN'] = X13
    C26['KLN'] = X14
    C21['KRP'] = X15
    C21['KLP'] = X16
    C21['KDP'] = X17
    C16['KRC'] = X18
    C16['KLC'] = X19
    C28['AONT'] = X20
    
    
    #########--------------------------纳入不再手动调的藻和水生植被
    C11['TMc1'] = 22
    C11['TMc2'] = 30
    C11['TMg1'] = 22
    C11['TMg2'] = 25
    C11['TMd1'] = 10
    C11['TMd2'] = 14
    C31['BMRg'] = 0.11
    C31['BMRc'] = 0.08
    C31['BMRd'] = 0.1
    C31['PMg'] = 1.5
    C31['PMc'] = 2.2
    C31['PMd'] = 2.2
    C32['WSd'] = 0.15
    C32['WSg'] = 0.25
    C32['WSc'] = 0.18
    C31['PRRc'] = 0.1
    C31['PRRd'] = 0.15
    C31['PRRg'] = 0.07
    C08['KHNc'] = 0.0005
    C08['KHNd'] = 0.009
    C08['KHNg'] = 0.007
    C08['KHPc'] = 0.00005
    C08['KHPd'] = 0.0009
    C08['KHPg'] = 0.0007
    Z02['IWD_KHNm'] = 0.01
    Z02['IWD_KHPm'] = 0.001
    Z02['IWD_KHNs'] = 0.01
    Z02['IWD_KHPs'] = 0.001
    Z03['IWD_DOPTm'] = 2
    Z08['VEGH'] = 1.9
    Z08['VEGBM'] = 0.7  
    Z08['VEGTM'] = 1.9
    Z08['KPHYTO'] = 200
    Z08['KBP'] = 150
    Z05['IWD_PCR'] = 0.0022
    Z07['IWD_PMm'] = 0.45
    Z07['IWD_BMRm'] = 0.016
    Z07['IWD_PRRm'] = 0.02
    Z07['IWD_SETM'] = 0.012
    Z03['IWD_TMm1'] = 20
    Z03['IWD_TMm2'] = 25
    Z03['IWD_KTG1m'] = 0.01

###-----------------------------------------------------------------------------------------------------------------------------------------------------
    with open(fileWENJIANpath + '\Input\wq3dwc.inp', 'w') as tf:
        tf.write(dict1['first part']),
        for k08 in C08:
            tf.write(str(C08[k08])+' ')
        tf.write('\n')
        for k09 in C09:
            tf.write(str(C09[k09])+' ')
        tf.write('\n')
        for k10 in C10:
            tf.write(str(C10[k10])+' ')
        tf.write('\n')
        for k11 in C11:
            tf.write(str(C11[k11])+' ')
        tf.write('\n')
        for k12 in C12:
            tf.write(str(C12[k12])+' ')
        tf.write('\n')
        for k13 in C13:
            tf.write(str(C13[k13])+' ')
        tf.write('\n')
        for k14 in C14:
            tf.write(str(C14[k14])+' ')
        tf.write('\n')
        for k15 in C15:
            tf.write(str(C15[k15])+' ')
        tf.write('\n')
        for k16 in C16:
            tf.write(str(C16[k16])+' ')
        tf.write('\n')
        for k17 in C17:
            tf.write(str(C17[k17])+' ')
        tf.write('\n')
        for k18 in C18:
            tf.write(str(C18[k18])+' ')
        tf.write('\n')
        for k19 in C19:
            tf.write(str(C19[k19])+' ')
        tf.write('\n')
        for k20 in C20:
            tf.write(str(C20[k20])+' ')
        tf.write('\n')
        for k21 in C21:
            tf.write(str(C21[k21])+' ')
        tf.write('\n')
        for k22 in C22:
            tf.write(str(C22[k22])+' ')
        tf.write('\n')
        for k23 in C23:
            tf.write(str(C23[k23])+' ')
        tf.write('\n')
        for k24 in C24:
            tf.write(str(C24[k24])+' ')
        tf.write('\n')
        for k25 in C25:
            tf.write(str(C25[k25])+' ')
        tf.write('\n')
        for k26 in C26:
            tf.write(str(C26[k26])+' ')
        tf.write('\n')
        for k27 in C27:
            tf.write(str(C27[k27])+' ')
        tf.write('\n')
        for k28 in C28:
            tf.write(str(C28[k28])+' ')
        tf.write('\n')
        for k29 in C29:
            tf.write(str(C29[k29])+' ')
        tf.write('\n')
        for k30 in C30:
            tf.write(str(C30[k30])+' ')
        tf.write('\n')
        for k31 in C31:
            tf.write(str(C31[k31])+' ')
        tf.write('\n')
        for k32 in C32:
            tf.write(str(C32[k32])+' ')
        tf.write('\n')
        tf.write(dict2['third part'])
    
    
    with open(fileWENJIANpath + '\Input\Peri_Macrophyte.inp', 'w') as gf:
        for l01 in Z01:
            gf.write(str(Z01[l01])+' ')
        gf.write('\n')
        for l02 in Z02:
            gf.write(str(Z02[l02])+' ')
        gf.write('\n')
        for l03 in Z03:
            gf.write(str(Z03[l03])+' ')
        gf.write('\n')
        for l04 in Z04:
            gf.write(str(Z04[l04])+' ')
        gf.write('\n')
        for l05 in Z05:
            gf.write(str(Z05[l05])+' ')
        gf.write('\n')
        for l06 in Z06:
            gf.write(str(Z06[l06])+' ')
        gf.write('\n')
        for l07 in Z07:
            gf.write(str(Z07[l07])+' ')
        gf.write('\n')
        for l08 in Z08:
            gf.write(str(Z08[l08])+' ')
        gf.write('\n')
    
    os.system( Iwindpath + '\Input\IWIND_LR_Competition.exe' + ' '+ Iwindpath + '\Input')

    ### 计算纳什系数的模块组，需要用到的有直接从软件导出的WQWCT文件和未加改动的observed文件
    obs_init = pd.read_table(os.path.join(path,'observed.inp'),sep='\s+',encoding='UTF-8')
    obs_init['Tbegin']  = '12/31/2017'
    obs_init['日期'] = pd.to_datetime(obs_init['日期'])
    obs_init['Tbegin'] = pd.to_datetime(obs_init['Tbegin'])
    obs_init['TIME'] =( obs_init['日期'] - obs_init['Tbegin']).dt.days
    obs = pd.DataFrame(obs_init, columns=['I','J','K','TIME','Chla','TP','NH4','TN','MP1','NO3','PO4'])
    obs_xyzt = pd.DataFrame(obs_init, columns=['I','J','K','TIME'])
    obs_k2 = obs.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']]
    obs_k1 = obs.loc[obs["K"] == 1, ['MP1']]
    result_init = pd.read_table(os.path.join(path,'Input','WQWCTS.OUT'),sep='\s+') 
    result_init['TIME'] = result_init['TIME'].astype(int)
    ###给不同的水质参数赋值
    result_init['Chla'] = result_init['CHC'] + result_init['CHD'] + result_init['CHG']
    result_init['TN'] = result_init['RON'] + result_init['LON'] + result_init['LDN']+ result_init['RDN'] + result_init['NH4'] + result_init['NO3'] + (result_init['CHC']*C09['CChlc']*C24['ANCc'] +result_init['CHD']*C09['CChld']*C24['ANCd']+result_init['CHG']*C09['CChlg']*C24['ANCg'])                                                                                                                                                             
    result_init['TP'] = result_init['ROP'] + result_init['LOP'] + result_init['LDP'] + result_init['RDP'] + result_init['PO4'] + ((result_init['CHC']*C09['CChlc']+result_init['CHD']*C09['CChld']+result_init['CHG']*C09['CChlg'])/C21['CPprm1'])
    result_sim = pd.DataFrame(result_init, columns=['I','J','K','TIME','Chla','TP','NH4','TN','MP1','NO3','PO4']) 
    result = pd.merge(obs_xyzt,result_sim,on=['I','J','K','TIME'], how='left')
    ###============================================================================================
    ###============================================================================================
    def plotvariable(variablename):
        listx = list(range(0,len(fendian_p1_simk2)));plt.figure(figsize=(30,25), dpi=80);ax1 = plt.subplot(541);ax1.set_title('S1');ax1.plot(listx,fendian_p1_simk2[variablename], color="r");ax1.scatter(listx,fendian_q1_obsk2[variablename], color="b");ax2 = plt.subplot(542);ax2.set_title('S2');ax2.plot(listx,fendian_p2_simk2[variablename], color="r");ax2.scatter(listx,fendian_q2_obsk2[variablename], color="b");ax3 = plt.subplot(543);ax3.set_title('S3');ax3.plot(listx,fendian_p3_simk2[variablename], color="r");ax3.scatter(listx,fendian_q3_obsk2[variablename], color="b");ax4 = plt.subplot(544);ax4.set_title('S4');ax4.plot(listx,fendian_p4_simk2[variablename], color="r");ax4.scatter(listx,fendian_q4_obsk2[variablename], color="b");ax5 = plt.subplot(545);ax5.set_title('S5');ax5.plot(listx,fendian_p5_simk2[variablename], color="r");ax5.scatter(listx,fendian_q5_obsk2[variablename], color="b");ax5 = plt.subplot(545);ax5.set_title('S5');ax5.scatter(listx,fendian_p5_simk2[variablename], color="r");ax5.scatter(listx,fendian_q5_obsk2[variablename], color="b");ax6 = plt.subplot(546);ax6.set_title('S6');ax6.plot(listx,fendian_p6_simk2[variablename], color="r");ax6.scatter(listx,fendian_q6_obsk2[variablename], color="b");ax7 = plt.subplot(547);ax7.set_title('S7');ax7.plot(listx,fendian_p7_simk2[variablename], color="r");ax7.scatter(listx,fendian_q7_obsk2[variablename], color="b");ax8 = plt.subplot(548);ax8.set_title('S8');ax8.plot(listx,fendian_p8_simk2[variablename], color="r");ax8.scatter(listx,fendian_q8_obsk2[variablename], color="b");ax9 = plt.subplot(549);ax9.set_title('S9');ax9.plot(listx,fendian_p9_simk2[variablename], color="r");ax9.scatter(listx,fendian_q9_obsk2[variablename], color="b");ax10 = plt.subplot(5,4,10);ax10.set_title('S10');ax10.plot(listx,fendian_p10_simk2[variablename], color="r");ax10.scatter(listx,fendian_q10_obsk2[variablename], color="b");ax11 = plt.subplot(5,4,11);ax11.set_title('S11');ax11.plot(listx,fendian_p11_simk2[variablename], color="r");ax11.scatter(listx,fendian_q11_obsk2[variablename], color="b");ax12 = plt.subplot(5,4,12);ax12.set_title('S12');ax12.plot(listx,fendian_p12_simk2[variablename], color="r");ax12.scatter(listx,fendian_q12_obsk2[variablename], color="b");ax13 = plt.subplot(5,4,13);ax13.set_title('S13');ax13.plot(listx,fendian_p13_simk2[variablename], color="r");ax13.scatter(listx,fendian_q13_obsk2[variablename], color="b");ax14 = plt.subplot(5,4,14);ax14.set_title('S14');ax14.plot(listx,fendian_p14_simk2[variablename], color="r");ax14.scatter(listx,fendian_q14_obsk2[variablename], color="b");ax15 = plt.subplot(5,4,15);ax15.set_title('S15');ax15.plot(listx,fendian_p15_simk2[variablename], color="r");ax15.scatter(listx,fendian_q15_obsk2[variablename], color="b");ax16 = plt.subplot(5,4,16);ax16.set_title('S16');ax16.plot(listx,fendian_p16_simk2[variablename], color="r");ax16.scatter(listx,fendian_q16_obsk2[variablename], color="b");ax17 = plt.subplot(5,4,17);ax17.set_title('S17');ax17.plot(listx,fendian_p17_simk2[variablename], color="r");ax17.scatter(listx,fendian_q17_obsk2[variablename], color="b");ax18 = plt.subplot(5,4,18);ax18.set_title('S18');ax18.plot(listx,fendian_p18_simk2[variablename], color="r");ax18.scatter(listx,fendian_q18_obsk2[variablename], color="b");ax19 = plt.subplot(5,4,19);ax19.set_title('S19');ax19.plot(listx,fendian_p19_simk2[variablename], color="r");ax19.scatter(listx,fendian_q19_obsk2[variablename], color="b");ax20 = plt.subplot(5,4,20);ax20.set_title('S20');ax20.plot(listx,fendian_p20_simk2[variablename], color="r");ax20.scatter(listx,fendian_q20_obsk2[variablename], color="b");
        tickstime = time.strftime('%Y_%m_%d_%H_%M',time.localtime(time.time()))
        plt.savefig(picturepath + "\\"+ variablename + "_"+ tickstime + ".png")
        return 
    def plotvariable_MP1(variablename):
        listx = list(range(0,len(fendian_p1_simk1)));plt.figure(figsize=(30,25), dpi=80);ax1 = plt.subplot(541);ax1.set_title('S1');ax1.plot(listx,fendian_p1_simk1[variablename], color="r");ax1.scatter(listx,fendian_q1_obsk1[variablename], color="b");ax2 = plt.subplot(542);ax2.set_title('S2');ax2.plot(listx,fendian_p2_simk1[variablename], color="r");ax2.scatter(listx,fendian_q2_obsk1[variablename], color="b");ax3 = plt.subplot(543);ax3.set_title('S3');ax3.plot(listx,fendian_p3_simk1[variablename], color="r");ax3.scatter(listx,fendian_q3_obsk1[variablename], color="b");ax4 = plt.subplot(544);ax4.set_title('S4');ax4.plot(listx,fendian_p4_simk1[variablename], color="r");ax4.scatter(listx,fendian_q4_obsk1[variablename], color="b");ax5 = plt.subplot(545);ax5.set_title('S5');ax5.plot(listx,fendian_p5_simk1[variablename], color="r");ax5.scatter(listx,fendian_q5_obsk1[variablename], color="b");ax5 = plt.subplot(545);ax5.set_title('S5');ax5.scatter(listx,fendian_p5_simk1[variablename], color="r");ax5.scatter(listx,fendian_q5_obsk1[variablename], color="b");ax6 = plt.subplot(546);ax6.set_title('S6');ax6.plot(listx,fendian_p6_simk1[variablename], color="r");ax6.scatter(listx,fendian_q6_obsk1[variablename], color="b");ax7 = plt.subplot(547);ax7.set_title('S7');ax7.plot(listx,fendian_p7_simk1[variablename], color="r");ax7.scatter(listx,fendian_q7_obsk1[variablename], color="b");ax8 = plt.subplot(548);ax8.set_title('S8');ax8.plot(listx,fendian_p8_simk1[variablename], color="r");ax8.scatter(listx,fendian_q8_obsk1[variablename], color="b");ax9 = plt.subplot(549);ax9.set_title('S9');ax9.plot(listx,fendian_p9_simk1[variablename], color="r");ax9.scatter(listx,fendian_q9_obsk1[variablename], color="b");ax10 = plt.subplot(5,4,10);ax10.set_title('S10');ax10.plot(listx,fendian_p10_simk1[variablename], color="r");ax10.scatter(listx,fendian_q10_obsk1[variablename], color="b");ax11 = plt.subplot(5,4,11);ax11.set_title('S11');ax11.plot(listx,fendian_p11_simk1[variablename], color="r");ax11.scatter(listx,fendian_q11_obsk1[variablename], color="b");ax12 = plt.subplot(5,4,12);ax12.set_title('S12');ax12.plot(listx,fendian_p12_simk1[variablename], color="r");ax12.scatter(listx,fendian_q12_obsk1[variablename], color="b");ax13 = plt.subplot(5,4,13);ax13.set_title('S13');ax13.plot(listx,fendian_p13_simk1[variablename], color="r");ax13.scatter(listx,fendian_q13_obsk1[variablename], color="b");ax14 = plt.subplot(5,4,14);ax14.set_title('S14');ax14.plot(listx,fendian_p14_simk1[variablename], color="r");ax14.scatter(listx,fendian_q14_obsk1[variablename], color="b");ax15 = plt.subplot(5,4,15);ax15.set_title('S15');ax15.plot(listx,fendian_p15_simk1[variablename], color="r");ax15.scatter(listx,fendian_q15_obsk1[variablename], color="b");ax16 = plt.subplot(5,4,16);ax16.set_title('S16');ax16.plot(listx,fendian_p16_simk1[variablename], color="r");ax16.scatter(listx,fendian_q16_obsk1[variablename], color="b");ax17 = plt.subplot(5,4,17);ax17.set_title('S17');ax17.plot(listx,fendian_p17_simk1[variablename], color="r");ax17.scatter(listx,fendian_q17_obsk1[variablename], color="b");ax18 = plt.subplot(5,4,18);ax18.set_title('S18');ax18.plot(listx,fendian_p18_simk1[variablename], color="r");ax18.scatter(listx,fendian_q18_obsk1[variablename], color="b");ax19 = plt.subplot(5,4,19);ax19.set_title('S19');ax19.plot(listx,fendian_p19_simk1[variablename], color="r");ax19.scatter(listx,fendian_q19_obsk1[variablename], color="b");ax20 = plt.subplot(5,4,20);ax20.set_title('S20');ax20.plot(listx,fendian_p20_simk1[variablename], color="r");ax20.scatter(listx,fendian_q20_obsk1[variablename], color="b");
        tickstime = time.strftime('%Y_%m_%d_%H_%M',time.localtime(time.time()))
        plt.savefig(picturepath + "\\"+ variablename + "_"+ tickstime + ".png")
        return 
    def lanlvgui_plot():
        fig,ax = plt.subplots(5,4,figsize=(30,24))
        jk = [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57]
        for i_chla,jjs in enumerate(jk):
            listx = list(range(0,73));
            ax[int(i_chla/4),int(i_chla%4)].stackplot(listx,pdchlajihe.iloc[:,jjs],pdchlajihe.iloc[:,jjs+1],pdchlajihe.iloc[:,jjs+2],colors=['cyan', 'yellow', 'green'],alpha=.6)    
            ax[int(i_chla/4),int(i_chla%4)].set_title('S'+str(i_chla))
            ax[int(i_chla/4),int(i_chla%4)].scatter(listx,fendianq_ob2.iloc[:,i_chla], color="r",s=15);
            ax2 = ax[int(i_chla/4),int(i_chla%4)].twinx()
            ax2.plot(listx,pdwendujihe.iloc[:,i_chla],'pink')
            tickstime = time.strftime('%Y_%m_%d_%H_%M',time.localtime(time.time()))
            fig.savefig(picturepath + '\lanlvgui_'+tickstime+'.png')
        return 
    ###=================
    point1 = result[result['I']==6]; point2 = result[result['I']==14]; point3 = result[result['I']==20]; point4 = result[result['I']==10]; point5 = result[result['I']==17]; point6 = result[result['I']==23]; point7 = result[result['I']==31]; point8 = result[result['J']==20]; point9 = result[result['I']==43]; point10 = result[result['I']==50]; point11 = result[result['I']==56]; point12 = result[result['J']==27]; point13 = result[result['I']==30]; point14 = result[result['I']==25]; point15 = result[result['I']==29]; point16 = result[result['I']==12]; point17 = result[result['I']==27]; point18 = result[result['I']==34]; point19 = result[result['I']==41]; point20 = result[result['I']==52]; 
    fendian_p1_simk2 = point1.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_p2_simk2 = point2.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_p3_simk2 = point3.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_p4_simk2 = point4.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_p5_simk2 = point5.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_p6_simk2 = point6.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_p7_simk2 = point7.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_p8_simk2 = point8.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_p9_simk2 = point9.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_p10_simk2 = point10.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_p11_simk2 = point11.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_p12_simk2 = point12.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_p13_simk2 = point13.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_p14_simk2 = point14.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_p15_simk2 = point15.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_p16_simk2 = point16.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_p17_simk2 = point17.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_p18_simk2 = point18.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_p19_simk2 = point19.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_p20_simk2 = point20.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];
    fendian_p1_simk1 = point1.loc[obs["K"] == 1, ['MP1']];fendian_p2_simk1 = point2.loc[obs["K"] == 1, ['MP1']];fendian_p3_simk1 = point3.loc[obs["K"] == 1, ['MP1']];fendian_p4_simk1 = point4.loc[obs["K"] == 1, ['MP1']];fendian_p5_simk1 = point5.loc[obs["K"] == 1, ['MP1']];fendian_p6_simk1 = point6.loc[obs["K"] == 1, ['MP1']];fendian_p7_simk1 = point7.loc[obs["K"] == 1, ['MP1']];fendian_p8_simk1 = point8.loc[obs["K"] == 1, ['MP1']];fendian_p9_simk1 = point9.loc[obs["K"] == 1, ['MP1']];fendian_p10_simk1 = point10.loc[obs["K"] == 1, ['MP1']];fendian_p11_simk1 = point11.loc[obs["K"] == 1, ['MP1']];fendian_p12_simk1 = point12.loc[obs["K"] == 1, ['MP1']];fendian_p13_simk1 = point13.loc[obs["K"] == 1, ['MP1']];fendian_p14_simk1 = point14.loc[obs["K"] == 1, ['MP1']];fendian_p15_simk1 = point15.loc[obs["K"] == 1, ['MP1']];fendian_p16_simk1 = point16.loc[obs["K"] == 1, ['MP1']];fendian_p17_simk1 = point17.loc[obs["K"] == 1, ['MP1']];fendian_p18_simk1 = point18.loc[obs["K"] == 1, ['MP1']];fendian_p19_simk1 = point19.loc[obs["K"] == 1, ['MP1']];fendian_p20_simk1 = point20.loc[obs["K"] == 1, ['MP1']];
    ## ======================================================================================
    qoint1 = obs[obs['I']==6]; qoint2 = obs[obs['I']==14]; qoint3 = obs[obs['I']==20]; qoint4 = obs[obs['I']==10]; qoint5 = obs[obs['I']==17]; qoint6 = obs[obs['I']==23]; qoint7 = obs[obs['I']==31]; qoint8 = obs[obs['J']==20]; qoint9 = obs[obs['I']==43]; qoint10 = obs[obs['I']==50]; qoint11 = obs[obs['I']==56]; qoint12 = obs[obs['J']==27]; qoint13 = obs[obs['I']==30]; qoint14 = obs[obs['I']==25]; qoint15 = obs[obs['I']==29]; qoint16 = obs[obs['I']==12]; qoint17 = obs[obs['I']==27]; qoint18 = obs[obs['I']==34]; qoint19 = obs[obs['I']==41]; qoint20 = obs[obs['I']==52]; 
    fendian_q1_obsk2 = qoint1.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_q2_obsk2 = qoint2.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_q3_obsk2 = qoint3.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_q4_obsk2 = qoint4.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_q5_obsk2 = qoint5.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_q6_obsk2 = qoint6.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_q7_obsk2 = qoint7.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_q8_obsk2 = qoint8.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_q9_obsk2 = qoint9.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_q10_obsk2 = qoint10.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_q11_obsk2 = qoint11.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_q12_obsk2 = qoint12.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_q13_obsk2 = qoint13.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_q14_obsk2 = qoint14.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_q15_obsk2 = qoint15.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_q16_obsk2 = qoint16.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_q17_obsk2 = qoint17.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_q18_obsk2 = qoint18.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_q19_obsk2 = qoint19.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];fendian_q20_obsk2 = qoint20.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']];
    fendian_q1_obsk1 = qoint1.loc[obs["K"] == 1, ['MP1']];fendian_q2_obsk1 = qoint2.loc[obs["K"] == 1, ['MP1']];fendian_q3_obsk1 = qoint3.loc[obs["K"] == 1, ['MP1']];fendian_q4_obsk1 = qoint4.loc[obs["K"] == 1, ['MP1']];fendian_q5_obsk1 = qoint5.loc[obs["K"] == 1, ['MP1']];fendian_q6_obsk1 = qoint6.loc[obs["K"] == 1, ['MP1']];fendian_q7_obsk1 = qoint7.loc[obs["K"] == 1, ['MP1']];fendian_q8_obsk1 = qoint8.loc[obs["K"] == 1, ['MP1']];fendian_q9_obsk1 = qoint9.loc[obs["K"] == 1, ['MP1']];fendian_q10_obsk1 = qoint10.loc[obs["K"] == 1, ['MP1']];fendian_q11_obsk1 = qoint11.loc[obs["K"] == 1, ['MP1']];fendian_q12_obsk1 = qoint12.loc[obs["K"] == 1, ['MP1']];fendian_q13_obsk1 = qoint13.loc[obs["K"] == 1, ['MP1']];fendian_q14_obsk1 = qoint14.loc[obs["K"] == 1, ['MP1']];fendian_q15_obsk1 = qoint15.loc[obs["K"] == 1, ['MP1']];fendian_q16_obsk1 = qoint16.loc[obs["K"] == 1, ['MP1']];fendian_q17_obsk1 = qoint17.loc[obs["K"] == 1, ['MP1']];fendian_q18_obsk1 = qoint18.loc[obs["K"] == 1, ['MP1']];fendian_q19_obsk1 = qoint19.loc[obs["K"] == 1, ['MP1']];fendian_q20_obsk1 = qoint20.loc[obs["K"] == 1, ['MP1']];
    ### 模拟值中的蓝藻、绿藻、硅藻
    lanzao_lvzao_guizao = pd.DataFrame(result_init, columns=['I','J','K','TIME','CHC','CHD','CHG']) 
    lanzao_lvzao_guizao_res = pd.merge(obs_xyzt,lanzao_lvzao_guizao,on=['I','J','K','TIME'], how='left')
    lanlvgui = lanzao_lvzao_guizao_res[lanzao_lvzao_guizao_res['K']==2].reset_index(drop=True)
    ### 模拟值中的温度
    wendu = pd.DataFrame(result_init, columns=['I','J','K','TIME','TEM']) 
    wendu_res = pd.merge(obs_xyzt,wendu,on=['I','J','K','TIME'], how='left')
    wendu_value = wendu_res[wendu_res['K']==2].reset_index(drop=True)
    chlapoint1 = lanlvgui[lanlvgui['I']==6].iloc[:,4:7].reset_index(drop=True);  chlapoint2 = lanlvgui[lanlvgui['I']==14].iloc[:,4:7].reset_index(drop=True);  chlapoint3 = lanlvgui[lanlvgui['I']==20].iloc[:,4:7].reset_index(drop=True);  chlapoint4 = lanlvgui[lanlvgui['I']==10].iloc[:,4:7].reset_index(drop=True);  chlapoint5 = lanlvgui[lanlvgui['I']==17].iloc[:,4:7].reset_index(drop=True);  chlapoint6 = lanlvgui[lanlvgui['I']==23].iloc[:,4:7].reset_index(drop=True);  chlapoint7 = lanlvgui[lanlvgui['I']==31].iloc[:,4:7].reset_index(drop=True);  chlapoint8 = lanlvgui[lanlvgui['J']==20].iloc[:,4:7].reset_index(drop=True);  chlapoint9 = lanlvgui[lanlvgui['I']==43].iloc[:,4:7].reset_index(drop=True);  chlapoint10 = lanlvgui[lanlvgui['I']==50].iloc[:,4:7].reset_index(drop=True);  chlapoint11 = lanlvgui[lanlvgui['I']==56].iloc[:,4:7].reset_index(drop=True);  chlapoint12 = lanlvgui[lanlvgui['J']==27].iloc[:,4:7].reset_index(drop=True);  chlapoint13 = lanlvgui[lanlvgui['I']==30].iloc[:,4:7].reset_index(drop=True);  chlapoint14 = lanlvgui[lanlvgui['I']==25].iloc[:,4:7].reset_index(drop=True);  chlapoint15 = lanlvgui[lanlvgui['I']==29].iloc[:,4:7].reset_index(drop=True);  chlapoint16 = lanlvgui[lanlvgui['I']==12].iloc[:,4:7].reset_index(drop=True);  chlapoint17 = lanlvgui[lanlvgui['I']==27].iloc[:,4:7].reset_index(drop=True);  chlapoint18 = lanlvgui[lanlvgui['I']==34].iloc[:,4:7].reset_index(drop=True);  chlapoint19 = lanlvgui[lanlvgui['I']==41].iloc[:,4:7].reset_index(drop=True);  chlapoint20 = lanlvgui[lanlvgui['I']==52].iloc[:,4:7].reset_index(drop=True);  pdchlajihe = pd.concat([chlapoint1,chlapoint2,chlapoint3,chlapoint4,chlapoint5,                         chlapoint6,chlapoint7,chlapoint8,chlapoint9,chlapoint10,                         chlapoint11,chlapoint12,chlapoint13,chlapoint14,chlapoint15,                         chlapoint16,chlapoint17,chlapoint18,chlapoint19,chlapoint20],axis=1)
    wendupoint1 = wendu_value[wendu_value['I']==6].iloc[:,4].reset_index(drop=True); wendupoint2 = wendu_value[wendu_value['I']==14].iloc[:,4].reset_index(drop=True); wendupoint3 = wendu_value[wendu_value['I']==20].iloc[:,4].reset_index(drop=True); wendupoint4 = wendu_value[wendu_value['I']==10].iloc[:,4].reset_index(drop=True); wendupoint5 = wendu_value[wendu_value['I']==17].iloc[:,4].reset_index(drop=True); wendupoint6 = wendu_value[wendu_value['I']==23].iloc[:,4].reset_index(drop=True); wendupoint7 = wendu_value[wendu_value['I']==31].iloc[:,4].reset_index(drop=True); wendupoint8 = wendu_value[wendu_value['J']==20].iloc[:,4].reset_index(drop=True); wendupoint9 = wendu_value[wendu_value['I']==43].iloc[:,4].reset_index(drop=True); wendupoint10 = wendu_value[wendu_value['I']==50].iloc[:,4].reset_index(drop=True); wendupoint11 = wendu_value[wendu_value['I']==56].iloc[:,4].reset_index(drop=True); wendupoint12 = wendu_value[wendu_value['J']==27].iloc[:,4].reset_index(drop=True); wendupoint13 = wendu_value[wendu_value['I']==30].iloc[:,4].reset_index(drop=True); wendupoint14 = wendu_value[wendu_value['I']==25].iloc[:,4].reset_index(drop=True); wendupoint15 = wendu_value[wendu_value['I']==29].iloc[:,4].reset_index(drop=True); wendupoint16 = wendu_value[wendu_value['I']==12].iloc[:,4].reset_index(drop=True); wendupoint17 = wendu_value[wendu_value['I']==27].iloc[:,4].reset_index(drop=True); wendupoint18 = wendu_value[wendu_value['I']==34].iloc[:,4].reset_index(drop=True); wendupoint19 = wendu_value[wendu_value['I']==41].iloc[:,4].reset_index(drop=True); wendupoint20 = wendu_value[wendu_value['I']==52].iloc[:,4].reset_index(drop=True); pdwendujihe = pd.concat([wendupoint1,wendupoint2,wendupoint3,wendupoint4,wendupoint5,                         wendupoint6,wendupoint7,wendupoint8,wendupoint9,wendupoint10,                         wendupoint11,wendupoint12,wendupoint13,wendupoint14,wendupoint15,                         wendupoint16,wendupoint17,wendupoint18,wendupoint19,wendupoint20],axis=1)
    fendianq_ob2 = pd.concat([fendian_q1_obsk2['Chla'].reset_index(drop=True),fendian_q2_obsk2['Chla'].reset_index(drop=True),fendian_q3_obsk2['Chla'].reset_index(drop=True),fendian_q4_obsk2['Chla'].reset_index(drop=True),fendian_q5_obsk2['Chla'].reset_index(drop=True),fendian_q6_obsk2['Chla'].reset_index(drop=True),fendian_q7_obsk2['Chla'].reset_index(drop=True),fendian_q8_obsk2['Chla'].reset_index(drop=True),fendian_q9_obsk2['Chla'].reset_index(drop=True),fendian_q10_obsk2['Chla'].reset_index(drop=True),fendian_q11_obsk2['Chla'].reset_index(drop=True),fendian_q12_obsk2['Chla'].reset_index(drop=True),fendian_q13_obsk2['Chla'].reset_index(drop=True),fendian_q14_obsk2['Chla'].reset_index(drop=True),fendian_q15_obsk2['Chla'].reset_index(drop=True),fendian_q16_obsk2['Chla'].reset_index(drop=True),fendian_q17_obsk2['Chla'].reset_index(drop=True),fendian_q18_obsk2['Chla'].reset_index(drop=True),fendian_q19_obsk2['Chla'].reset_index(drop=True),fendian_q20_obsk2['Chla'].reset_index(drop=True)],axis=1);
    plotvariable('Chla') ;plotvariable('TP') ;plotvariable('NH4') ;plotvariable('TN') ;plotvariable('NO3') ;plotvariable('PO4') ;plotvariable_MP1('MP1') ;
    lanlvgui_plot()
    ###============================================================================================
    ###============================================================================================
    ###============================================================================================
    ###============================================================================================
    ###============================================================================================
    ###============================================================================================
    ###============================================================================================
    ## 继续正常的操作
    sim_k2 = result.loc[obs["K"] == 2, ['Chla','TP','NH4','TN','NO3','PO4']]
    sim_k1 = result.loc[obs["K"] == 1, ['MP1']]
    top_k2 = np.sum(np.square(np.log10(obs_k2.values) - np.log10(sim_k2.values)),axis=0)
    top_k1 = np.sum(np.square(obs_k1.values - sim_k1.values),axis=0)
    obs_k1_mean = np.mean(obs_k1.values,axis=0)
    obs_k2_mean = np.mean(np.log10(obs_k2.values),axis=0)
    bottom_k2 = np.sum(np.square(np.log10(obs_k2.values) - obs_k2_mean),axis=0)
    bottom_k1 = np.sum(np.square(obs_k1.values - obs_k1_mean),axis=0)
    nse_k2 = 1-top_k2/bottom_k2
    nse_k1 = 1-top_k1/bottom_k1
    nash = (np.sum(nse_k2) + nse_k1)/7
    if np.isnan(nash[0]):
        nash[0] = -2000
    ###============================================================================================
    ###============================================================================================
    ###============================================================================================
    ###============================================================================================
    ###============================================================================================
    ###============================================================================================
    ###============================================================================================
    ## 继续正常的操作
    ####======================
    
    with open(withopen_nash,'a') as f1:
                f1.write(str(nash[0])+'\n')
    with open(withopen_seven,'a') as f2:
                f2.write(str(np.around(nse_k2,3))),f2.write('\t'),f2.write(str(np.around(nse_k1,3))),f2.write('\n')
                
    return nash[0]        


#######定义结束黑箱模型


black_box_function(X01,X02,X03,X04,X05,X06,X07,X08,X09,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20)

pbounds = { 'X01':(0.01 , 0.1),
            'X02':(0.01 , 0.1),
            'X03':(0.05 , 0.5),
            'X04':(2 , 7),
            'X05':(0.1 , 0.35),
            'X06':(0.03 , 1),
            'X07':(0.008, 0.02),
            'X08':(0.1, 0.2),
            'X09':(0.1 , 1),
            'X10':(0.1  , 1),
            'X11':(0.1  , 1),
            'X12':(0.01 , 0.2),
            'X13':(0.0001 , 0.02),
            'X14':(0.01 , 0.3),
            'X15':(0.0001 , 0.02),
            'X16':(0.01 , 0.3),
            'X17':(0.01 , 0.3),
            'X18':(0.0001 , 0.02),
            'X19':(0.01 , 0.3),
            'X20':(1 , 5)
}
    
optimizer = BayesianOptimization( f = black_box_function, pbounds = pbounds, random_state = 1)
##指定初始探测值probe

optimizer.probe(params={
            'X01':0.069,
            'X02':0.069,
            'X03':0.3,
            'X04':2,
            'X05':0.15,
            'X06':0.9,
            'X07':0.0015,
            'X08':0.06,
            'X09':0.3,
            'X10':0.3,
            'X11':1,
            'X12':0.06,
            'X13':0.001,
            'X14':0.04,
            'X15':0.001,
            'X16':0.04,
            'X17':0.07,
            'X18':0.001,
            'X19':0.04,
            'X20':4.33

},lazy=True)


##导入load函数
load_logs(optimizer, logs=["C:\\Users\\PKUWSL-JLin\\Desktop\\AUTO\\RESULT\\logs_NASH_Out1_iterB.json"])            

logger = JSONLogger(path= 'C:\\Users\\PKUWSL-JLin\\Desktop\\AUTO\\RESULT\\logs_NASH_Out1_iterD.json'  )
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

optimizer.maximize(init_points=0, n_iter=300 ,acq='ucb',kappa=0.5)

##CRAYY  ON  黑箱模型


