# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:35:47 2022

@author: r1vog
"""

import pandas as pd
import numpy as np
import time
import csv
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.integrate import quad
zeitanfang = time.time()


"""
Parameters
"""
Rooms = (1,2,3,4,5,6)           #Number of rooms, for following calculations

ap_tot_20 = (51632,118577,223316,211098,101379,58025)           #Number of apartments in Zürich with 1,2,3,4,5,6 Rooms
ap_tot_20_ch = (295847,679360,1253156,1276309,696445,436057)    #Number of apartments in Switzerland with 1,2,3,4,5,6 Rooms
ap_inhab_20 = (40421,106332,207580,198945,96525,54533)          #Number of inhabited apartments in Zürich with 1,2,3,4,5,6 Rooms
ap_unused_20 = np.subtract(ap_tot_20,ap_inhab_20)               #Number of unused apartments in Zürich with 1,2,3,4,5,6 Rooms

ap_tot_20_sum = sum(ap_tot_20)          #Total number of apartments in Zürich 
ap_tot_20_sum_ch = sum(ap_tot_20_ch)    #Total number of apartments in Schwitzerland 
ap_inhab_20_sum = sum(ap_inhab_20)      #Total number of inhabited apartments in Zürich 
ap_unused_sum = sum(ap_unused_20)       #Total number of unused apartments in Zürich 

ap_size = 97.2      #average size of apartment in zürich, m2
ap_size_ch = 99     #average size of apartment in switzerland, m2
ap_rooms = 3.6      #average rooms per HABITED apartment in zürich

Living_density_20 = 0.61 #Habitants per room in Zürich
Living_density_l = 0.5 #Habitants per room in Zürich
Living_density_u = 0.73 #Habitants per room in Zürich

Living_area_20 = 45.5 #area per person, m2
Living_area_l = 25.5 #area per person, m2
Living_area_u = 65.5 #area per person, m2

#Dem_rooms_20 = 7962

Goal_y = 2050 #goal year
Start_y = 2021 #Start year
Ref_y = 2020 #Reference Year

Population_20=1551342 #Zürich population in 2018
Population_50=2000000 #Zürich population in 2050
Population_50_CH = 10440600
Pop = np.linspace(Population_20,Population_50,Goal_y+2-Start_y) #Jährliche Population bis 2050 bei linarer zunahme

Ren_rate_20 = 1/100 #renovation rate

Em_emb_new = 696            #emission embodied new building, kgCO2eq/m2
Em_emb_new_l = -400         #emission embodied new building, kgCO2eq/m2/a lower bound
Em_emb_new_u = 900          #emission embodied new building, kgCO2eq/m2/a upper bound

Em_Emb_ren_20 = 440         #emission embodied renovation, kgCO2eq/m2
Em_Emb_ren_l = -200         #emission embodied renovation, kgCO2eq/m2
Em_Emb_ren_u = 200          #emission embodied renovation, kgCO2eq/m2

Em_Dem = 80                #emissions demolishion, kgCO2eq/m2

Em_Op_ren_20 = 5.8          #Emission operational renovated building in 2020, kgCO2eq/m2/a
Em_Op_ren_goal = 3.5

Em_op_new_20 = 3.5      #emissions operational 2020, kgCO2eq/m2/a
Em_op_new_goal = 3.5        #emissions operational 2020, kgCO2eq/m2/a

Em_Op_old_20 = 15.5  #kgCO2/m2/y
Em_Op_old_all_ren = Em_op_new_20

Rooms_Total_20 = sum(np.multiply(ap_tot_20, Rooms))             #Total number of Rooms in Zürich 
#Rooms_Total_20 = ap_tot_20_sum*ap_rooms
#Rooms_Total_20 = 2805956
Rooms_dem = 7962
#Rooms_Inhabited_20 = sum(np.multiply(ap_inhab_20, Rooms))       #Total number of inhabited Rooms in Zürich 
Rooms_Inhabited_20 = ap_inhab_20_sum*ap_rooms
Rooms_Inhabited_20 = Population_20/Living_density_20      #Total number of inhabited Rooms in Zürich 
#Rooms_Unused_20 = sum(np.multiply(ap_unused_20, Rooms))         #Total number of unused Rooms in Zürich
Rooms_Unused_20 = Rooms_Total_20-Rooms_Inhabited_20         #Total number of unused Rooms in Zürich
Rooms_Unused_l = 0                                         #Total number of unused Rooms in Zürich in 2050 lower limit
Rooms_Unused_u = Rooms_Unused_20*2                              #Total number of unused Rooms in Zürich in 2050 upper limit


# Erstellt array mit lower und upper bound
Em_emb_new_p = np.linspace(Em_emb_new_l,Em_emb_new_u,5)     #emission embodied, kgCO2eq/m2/a parametric range
Em_emb_ren_p = np.linspace(Em_Emb_ren_l,Em_Emb_ren_u,5)
Rooms_Unused_p = np.round(np.linspace(Rooms_Unused_l,Rooms_Unused_u, 5),0)
Rooms_Unused_p2 = np.round(Rooms_Unused_p/1000,0)
Living_area_p = np.linspace(Living_area_l, Living_area_u, 5)
Living_density_p = np.round(np.linspace(Living_density_l, Living_density_u,7),3)
Ren_share_p = np.linspace(0.85,0.99,5)
t_p = np.linspace(10,30,5)

New_cons_p = np.linspace(0,1500000,5)
New_cons_p2 = New_cons_p/1000000
Em_op_old_tot_p = np.linspace(0,20,5)
Em_op_ren_tot_p = np.linspace(0,10,5)
Em_emb_ren_tot_p = np.linspace(0,30,5)
Em_op_new_tot_p = np.linspace(0,5,5)
Em_emb_new_tot_p = np.linspace(0,60,5)
Em_dem_tot_p = np.linspace(0,2,5)
Em_tot_tot_p = np.round(np.linspace(15,100,5),1)

"""
Calculation parameters
"""
#Building surface in Switzerland and in Zürich
S_OLD_20 = ap_size*ap_tot_20_sum                #surface of existing buildings in Zürich (residential)
S_OLD_20_ch = ap_size_ch*ap_tot_20_sum_ch       #surface of existing buildings in Switzerland (residential)

#Calculation of the Carbon budget for the construction sector in zürich 
CB_CH_1_5_tot = 476                 #Total carbon budget switzerland for 1.5 degree warming, MtCO2eq.
CB_CH_2_tot = 1326                  #Total carbon budget switzerland for 2 degree warming, MtCO2eq.

CB_CH_1_5_op_buil_18 = CB_CH_1_5_tot*0.21                          #Carbon budget buildings operational switzerland for 1.5 degree warming, MtCO2eq.
CB_CH_2_op_buil_18 = CB_CH_2_tot*0.21                              #Carbon budget buildings operational switzerland for 2 degree warming, MtCO2eq.

#CB_CH_1_5_op_buil = (-CB_CH_1_5_op_buil_18*2/32/32*2 + CB_CH_1_5_op_buil_18*2/32)*30/2                         #Carbon budget buildings operational switzerland for 1.5 degree warming, MtCO2eq.
#CB_CH_2_op_buil = (-CB_CH_2_op_buil_18*2/32/32*2 + CB_CH_2_op_buil_18*2/32)*30/2                             #Carbon budget buildings operational switzerland for 2 degree warming, MtCO2eq.

CB_CH_1_5_op_buil = CB_CH_1_5_op_buil_18-7.69-7.12-3.51-3.26            #Carbon budget residential buildings operational switzerland for 1.5 degree warming, MtCO2eq.
CB_CH_2_op_buil = CB_CH_2_op_buil_18-7.69-7.12-3.51-3.26                #Carbon budget residential buildings operational switzerland for 2 degree warming, MtCO2eq.

CB_RBS_CH_1_5_op = CB_CH_1_5_op_buil*16.4/(16.4+7.5)               #Carbon budget residential buildings operational switzerland for 1.5 degree warming, MtCO2eq.
CB_RBS_CH_2_op = CB_CH_2_op_buil*16.4/(16.4+7.5)                   #Carbon budget residential buildings operational switzerland for 2 degree warming, MtCO2eq.

CB_CH_1_5_emb_dom = (CB_CH_1_5_tot*0.24-11.20-10.74)*0.4*0.3                  #Carbon budget buildings domestic embodied emissions switzerland for 1.5 degree warming, MtCO2eq.
CB_CH_2_emb_dom = (CB_CH_2_tot*0.24-11.20-10.74)*0.4*0.3                      #Carbon budget buildings domestic embodied emissions switzerland for 2 degree warming, MtCO2eq.

CB_CH_1_5_emb_imp = (CB_CH_1_5_tot*0.24-11.20-10.74)*0.4*0.3/0.3*0.7          #Carbon budget buildings imported embodied emissions switzerland for 1.5 degree warming, MtCO2eq.
CB_CH_2_emb_imp = (CB_CH_2_tot*0.24-11.20-10.74)*0.4*0.3/0.3*0.7              #Carbon budget buildings imported embodied emissions switzerland for 2 degree warming, MtCO2eq.

CB_RBS_CH_1_5_emb = (CB_CH_1_5_emb_dom+CB_CH_1_5_emb_imp)*(549-542)/(812-802)       #Carbon budget residential buildings embodied emissions switzerland for 1.5 degree warming, MtCO2eq.
CB_RBS_CH_2_emb = (CB_CH_2_emb_dom+CB_CH_2_emb_imp)*(549-542)/(812-802)             #Carbon budget residential buildings embodied emissions switzerland for 2 degree warming, MtCO2eq.

CB_RBS_CH_1_5 = CB_RBS_CH_1_5_op+CB_RBS_CH_1_5_emb              #Carbon budget residential building sector Schweiz
CB_RBS_CH_2 = CB_RBS_CH_2_op+CB_RBS_CH_2_emb                    #Carbon budget residential building sector Schweiz

CB_RBS_ZH_1_5 = CB_RBS_CH_1_5*Population_50/Population_50_CH        #Carbon budget residential building sector Schweiz
CB_RBS_ZH_2 = CB_RBS_CH_2*Population_50/Population_50_CH            #Carbon budget residential building sector Schweiz
 


"""
Functions
"""
def warming_degree(Em_tot_f):
    a = 0.5/(CB_RBS_ZH_2-CB_RBS_ZH_1_5)
    b = 1.5 - a*CB_RBS_ZH_1_5
    return a*Em_tot_f + b

wd_p = np.round(warming_degree(Em_tot_tot_p),1)

#Function renovation rate, exponential growing from 2020-2050
B_ren = Ren_rate_20
#t = 10
#Tau = 2
#E_ren = (P_ren_tot+B_ren*(Tau*np.exp(-t/Tau)-Tau))/(t+Tau*np.exp(-t/Tau)-Tau)
#E_ren = (P_ren_tot-t*B_ren)/(t**2)*2 # a von ax + b 
def Func_ren_rate(t,E_ren_f,t_f):
    t2 = t - 0.5
    if E_ren_f*t2 + B_ren > 0 and t <= t_f: 
        return E_ren_f*t2 + B_ren
    #if E_ren*t2 + B_ren <= 0 or t > t_f:
        #return E_ren_f*(1-np.exp(-t/Tau))+B_ren*np.exp(-t/Tau)
    else:
        return 0

#I1 = quad(Func_ren_rate, 0, t, args=(E_ren,t))



#Function demolition rate, linear growing from 2020-2050
B_dem = Rooms_dem/Rooms_Total_20
#a_dem = (P_dem_tot-t*B_dem)/(t**2)*2 # a von ax + b 
def Func_dem_rate(t,P_dem_tot_f,t_f):
    t2 = t-0.5
    a_dem = (P_dem_tot_f-(t_f)*B_dem)/((t_f)**2)*2 # a von ax + b 
    if P_dem_tot_f < B_dem*t_f/2:
        t_x = P_dem_tot_f/B_dem*2
        a_dem_x = -B_dem/t_x        
        if t - t_x <= 0:
            return a_dem_x*t2 + B_dem
        if t - t_x < 1:
            return (a_dem_x*(t-1+(t_x-(t-1))/2) + B_dem)*(t_x-(t-1))
        if t >= t_x:
            return 0
        
    if a_dem*t2 + B_dem > 0 and t<=t_f: 
        return a_dem*t2 + B_dem
    else:
        return 0
#I2 = quad(Func_dem_rate, 0, t,args=(a_dem,10))

##linear development emissions according stock renovated/demolished
def lin_development_em_stock(Em_value_20, Em_value_goal, Sur_old_f):  
    Em_y = (Em_value_20 - (Em_value_20-Em_value_goal)*(S_OLD_20-Sur_old_f)/S_OLD_20)
    return Em_y

#Linearisieren, Wert in Jahr xy
def linearise (year_i, year_Goal, value_2020, value_Goal): 
    return ((value_Goal-value_2020)/(year_Goal-Ref_y))*(year_i - Ref_y)+value_2020 

#Jählriche Zu/abnahme bei Linearem Verlauf  
def linearise_delta_y (year_Goal, value_2020, value_Goal):
    return ((value_2020-value_Goal)/(year_Goal-Ref_y)) 


def Calc_func(Living_density_f,Rooms_Unused_f,Living_area_f,Em_emb_new_f,Ren_share_f,t_f,Em_Emb_ren_f):
    Sur_dem = 0
    Sur_ren = 0
    Sur_old = S_OLD_20
    
    Em_Op_old_tot = 0
    Em_Op_ren_tot = 0 
    Em_Emb_ren_tot = 0
    Em_Dem_tot = 0
    Em_Op_new_tot = 0
    Em_Emb_new_tot = 0
    Em_tot = 0
    Em_Op_ren_y = 0
    Em_Op_new_y = 0
    
    Em_Op_old_tot_y = np.array([])
    Em_Op_ren_tot_y = np.array([]) 
    Em_Emb_ren_tot_y = np.array([])
    Em_Dem_tot_y = np.array([])
    Em_Op_new_tot_y = np.array([])
    Em_Emb_new_tot_y = np.array([])
    Em_tot_y = np.array([])
    Pers_new_build_tot_y = np.array([])
    Rooms_new_y = np.array([])
    Const_new_tot_y = np.array([])
    
    Rooms_dem = 0
    Rooms_new = 0
    Rooms_avail = Rooms_Inhabited_20
    
    Pers_new_build_tot = 0
    
    P_ren_tot = Ren_share_f
    #Em_Dem_tot = (1-P_ren_tot)*S_OLD_20*Em_Dem/1e9

    P_dem_tot =  round(1-P_ren_tot,4)
    E_ren = (P_ren_tot-t_f*B_ren)/(t_f**2)*2

    Dem_rate_tot = 0
    Ren_rate_tot = 0
    
    for y in range(Start_y,Goal_y+1):
        year = y - Start_y +1   #number of year, starts with 1
        
        Dem_rate_y = B_dem #Func_dem_rate(year,P_dem_tot,t_f)
        Dem_apart_y = Dem_rate_y*ap_tot_20_sum
        Ren_rate_y = B_ren #Func_ren_rate(year,E_ren,t_f)       
        
        Sur_ren += S_OLD_20*Ren_rate_y
        Sur_dem += Dem_rate_y*S_OLD_20
        Sur_old_prev = Sur_old
        Sur_old -= (S_OLD_20*Ren_rate_y + Dem_rate_y*S_OLD_20)
        Sur_old = round(Sur_old,6)
        Sur_old_avg = (Sur_old+Sur_old_prev)/2
        
        Living_density_y = linearise(y, Goal_y, Living_density_20, Living_density_f)
        Living_area_y = linearise(y, Goal_y,Living_area_20,Living_area_f) 
        Em_op_ren_y = lin_development_em_stock(Em_Op_ren_20,Em_Op_ren_goal,Sur_old_avg)
        Em_op_new_y = linearise(y, Goal_y,Em_op_new_20,Em_op_new_goal)
        Em_Op_old_y = lin_development_em_stock(Em_Op_old_20,Em_op_new_20,Sur_old_avg) 
        Em_emb_new_y = linearise(y, Goal_y,Em_emb_new,Em_emb_new_f)
        Em_emb_ren_y = lin_development_em_stock(Em_Emb_ren_20,Em_Emb_ren_f,Sur_old_avg)
        Rooms_reoccu_y = linearise(y,Goal_y, 0, (Rooms_Unused_20-Rooms_Unused_f)/30*2)
        Dem_rate_tot += Dem_rate_y
        Ren_rate_tot += Ren_rate_y
                
        Em_Op_old_tot_y = np.append(Em_Op_old_tot_y,Em_Op_old_y*Sur_old_avg/1e9)
        Em_Op_ren_tot_y = np.append(Em_Op_ren_tot_y,Em_Op_ren_y + Em_op_ren_y*S_OLD_20*Ren_rate_y*0.5/1e9)  #times 0.5 because renovation activities is consideret constant during year 
        Em_Op_ren_y += Em_op_ren_y*S_OLD_20*Ren_rate_y/1e9
        Em_Emb_ren_tot_y = np.append(Em_Emb_ren_tot_y,Ren_rate_y*S_OLD_20*Em_emb_ren_y/1e9)
        Em_Dem_tot_y = np.append(Em_Dem_tot_y,S_OLD_20*Dem_rate_y*Em_Dem/1e9)
        
        Rooms_dem = Dem_apart_y*ap_rooms
        Rooms_avail +=  (Rooms_new-Rooms_dem + Rooms_reoccu_y)
        Hab_poss = Rooms_avail*Living_density_y
        Hab_new_ap = Pop[year] - Hab_poss
        if Hab_new_ap > 0:
            Pers_new_build_tot_y = np.append(Pers_new_build_tot_y,Hab_new_ap)
            Rooms_new = Hab_new_ap/Living_density_y
            Rooms_new_y = np.append(Rooms_new_y,Rooms_new)
            Construction_new = Hab_new_ap*Living_area_y
            Const_new_tot_y = np.append(Const_new_tot_y,Construction_new)
            Em_Emb_new_tot_y = np.append(Em_Emb_new_tot_y,Construction_new*Em_emb_new_y/1e9)
            Em_Op_new_tot_y = np.append(Em_Op_new_tot_y,Em_Op_new_y + Construction_new*Em_op_new_y*0.5/1e9)
            Em_Op_new_y += Construction_new*Em_op_new_y/1e9
            
            
        else:
            Em_Emb_new_tot_y = np.append(Em_Emb_new_tot_y,0)
            Em_Op_new_tot_y = np.append(Em_Op_new_tot_y,0)
            Const_new_tot_y = np.append(Const_new_tot_y,0)
        
        Em_tot_y = np.append(Em_tot_y,Em_Op_old_tot_y[-1] + Em_Op_ren_tot_y[-1] + Em_Emb_ren_tot_y[-1] + Em_Op_new_tot_y[-1] + Em_Emb_new_tot_y[-1] + Em_Dem_tot_y[-1])
    
    Em_Op_old_tot = sum(Em_Op_old_tot_y)
    Em_Op_ren_tot = sum(Em_Op_ren_tot_y)
    Em_Emb_ren_tot = sum(Em_Emb_ren_tot_y)
    Em_Op_new_tot = sum(Em_Op_new_tot_y)
    Em_Emb_new_tot = sum(Em_Emb_new_tot_y)
    Em_Dem_tot = sum(Em_Dem_tot_y)
    Em_tot = sum(Em_tot_y)
    Pers_new_build_tot = sum(Pers_new_build_tot_y)
    Const_new_tot = sum(Const_new_tot_y)

    return Em_Op_old_tot,Em_Op_ren_tot, Em_Op_new_tot, Em_Emb_ren_tot,Em_Emb_new_tot,Em_Dem_tot,Em_tot,Pers_new_build_tot,Em_Op_old_tot_y,Em_Op_ren_tot_y,Em_Op_new_tot_y,Em_Emb_ren_tot_y,Em_Emb_new_tot_y,Em_Dem_tot_y,Em_tot_y,Pers_new_build_tot_y,Rooms_new_y,Const_new_tot

a = Calc_func(0.61, 100000, 25, 0, 0.9,30,-200)
#b = Calc_func(0.76, 0, 25, 0, 0.8,30,200)


BaU_res = Calc_func(Living_density_20, Rooms_Unused_20, Living_area_20, Em_emb_new, 0.95,30,Em_Emb_ren_20)
print('Warming potential =', warming_degree(BaU_res[6]))

#Plot bar plot
category_colors = plt.get_cmap('coolwarm_r')(np.linspace(0.05, 0.95, 6))
size_font = 13
res_val = list(BaU_res[0:6])
res_name = ['∑ Em. op. old','∑ Em. op. ren.','∑ Em. op. new','∑ Em. emb. ren.','∑ Em. emb. new','∑ Em. dem.']
fig, ax = plt.subplots(dpi = 500) 
plt.rc('xtick', labelsize=size_font) 
plt.rc('ytick', labelsize=size_font)
ax = fig.add_axes([0,0,1,1])
ax.bar(res_name,res_val,color = category_colors, edgecolor = 'black')
yticks = ax.get_yticks()[0:-1]
ylabel1 = []
ylabel2 = []
for z in range(0,len(yticks)):
    ylabel1.append(f'{int(yticks[z])}')
    ylabel2.append(f'{round(warming_degree(yticks[z]),2)}')
ax.set_yticks(yticks)
ax.set_yticklabels(ylabel1, fontsize = size_font)
plt.ylabel("Cumulative emissions [MtCo$_2$eq]", fontsize = size_font)
ax2 = ax.twinx()
ax2.bar(res_name,res_val,color = category_colors, edgecolor = 'black')
ax2.set_yticks(yticks)
ax2.set_yticklabels(ylabel2, fontsize = size_font)
ax.yaxis.grid(linewidth = 0.6, color = 'black', linestyle = 'dotted' )
plt.ylabel("Warming Potential [℃]", fontsize = size_font)
ax.tick_params(axis='x', labelrotation = 45)
percentage = list(np.round(res_val/sum(res_val)*100,1))
xaxis = ax.get_xticks()
for xz in range(0,len(xaxis)):
    ax2.text(xaxis[xz],res_val[xz] + 0.1,'%s %s' % (percentage[xz],'%'), color='black', ha = 'center')
plt.title('Cumulative emissions business as usual', fontsize = size_font)
plt.show()

#Plot pie chart total
fig1, ax1 = plt.subplots(dpi=500)
ax1.pie(res_val, autopct='%1.1f%%',startangle=-90,colors = category_colors,wedgeprops = {"edgecolor" : "black",'linewidth': 0.4,'antialiased': True},explode = [0,0.4,0.48,0,0,0])
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.legend(res_name,
          title="Emissions",
          loc="center left",
          bbox_to_anchor=(0.85, 0, 0.5, 1))
plt.tight_layout()
plt.show()



#plot emissions total
fig, ax = plt.subplots(dpi = 500)
ax.plot(np.arange(2021,2051,1),BaU_res[14],label = 'Yearly emissions') 
ax2 = ax.twinx()
ax2.plot(np.arange(2021,2051,1),warming_degree(np.cumsum(BaU_res[14])), color = 'red',linestyle = 'dotted', label = 'Warming potential')
plt.xlabel('Years',fontsize = size_font)
ax.set_ylabel('Yearly emissions [MtCO$_2$eq]',fontsize = size_font)
ax2.set_ylabel('Warming Potential [℃]',fontsize = size_font)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=4,fontsize = 13)
ax.set_xlabel('Year',fontsize = size_font)
ax.set_ylim([0, 2.2])
plt.title('Yearly total emissions business as usual', fontsize = size_font)
plt.show()

#Plot pie chart first year
res_val2 = [BaU_res[8][0],BaU_res[9][0],BaU_res[10][0],BaU_res[11][0],BaU_res[12][0],BaU_res[13][0]]
fig2, ax2 = plt.subplots(dpi=500)
ax2.pie(res_val2, autopct='  %1.1f%%',startangle=-90,colors = category_colors,wedgeprops = {"edgecolor" : "black",'linewidth': 0.4,'antialiased': True},explode = [0,0,0.1,0,0,0])
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax2.legend(res_name,
          title="Emissions",
          loc="center left",
          bbox_to_anchor=(0.85, 0, 0.5, 1))
plt.tight_layout()
plt.show()
