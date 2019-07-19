#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import numpy as np
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import cam_math
import math
import cam_stats
from numba import jit
from timeit import default_timer as timer
import os
ec = 1.6                # Соотношение координат области ожидания к bounding_box
size_ec = 1.2           # Соотношение размеров области ожидания к bounding_box
forget_rate = 500       # Количество циклов, необходимых для забывания объекта

# Размер равен количеству распознанных лиц
bound = []              # Массив bounding_box  [x, y, width, height]
ex_bound = []           # Массив областей ожидания
active_bound = []       # Буленовский массив, отражает активные/выключенные bounding_box
count_bound = []        # Массив чисел, для забывания объекта
bound_size_minmax = []
bound_coord_minmax = []

# Размер равен количеству распознанных лиц
active_time = []
first_active = []
last_active = []
looking_time = []
screen_time = []
frame_list = []

# Размер равен счетчику cycle_numb
stats_log = []
look_log = []
time_log = []

# Размер равен количеству распознанных лиц
another_count = []
another_numb = []
time_first_seen = []
time_last_seen = []

global_time = 0
cycle_numb = 0

@jit
def unity_minmaxes(coord_1, coord_2, size_1, size_2):
    if (coord_1[0][0] > coord_2[0][0]):
        coord_1[0][0] = coord_2[0][0]
    if (coord_1[0][1] < coord_2[0][1]):
        coord_1[0][1] = coord_2[0][1]
    if (coord_1[1][0] > coord_2[1][0]):
        coord_1[1][0] = coord_2[1][0]
    if (coord_1[1][1] < coord_2[1][1]):
        coord_1[1][1] = coord_2[1][1]
        
    if (size_1[0][0] > size_2[0][0]):
        size_1[0][0] = size_2[0][0]
    if (size_1[0][1] < size_2[0][1]):
        size_1[0][1] = size_2[0][1]
    if (size_1[1][0] > size_2[1][0]):
        size_1[1][0] = size_2[1][0]
    if (size_1[1][1] < size_2[1][1]):
        size_1[1][1] = size_2[1][1]    
        
        
@jit
def check_minmaxes(bouding, bound_size, bound_coord):
    x, y, sx, sy = bouding[0], bouding[1], bouding[2], bouding[3]
    if (sx > bound_size[0][1]):
        bound_size[0][1] = sx
    if (sx < bound_size[0][0]):
        bound_size[0][0] = sx   
    if (sy > bound_size[1][1]):
        bound_size[1][1] = sy
    if (sy < bound_size[1][0]):
        bound_size[1][0] = sy 
        
    if (x + sx > bound_coord[0][1]):
        bound_coord[0][1] = x + sx
    if (x < bound_coord[0][0]):
        bound_coord[0][0] = x
    if (y + sy > bound_coord[1][1]):
        bound_coord[1][1] = y + sy
    if (y < bound_coord[1][0]):
        bound_coord[1][0] = y    

@jit
def draw_basics(frame, bounding_box, uu, keypoints, color):
    angle = cam_math.get_angle(keypoints)
    x_angle = angle[0]
    y_angle = angle[1]
    #Прямоугольник лица человека
    cv2.rectangle(frame,
        (bounding_box[0], bounding_box[1]),
        (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
        color, 2)

    cv2.putText(frame, str(uu), (bounding_box[0] + 2, bounding_box[1] + 14), cv2.FONT_HERSHEY_PLAIN,
        1, (0, 155, 255), 1)
        
    #Указатель направления взгляда
    look_x = bounding_box[0] + bounding_box[2] // 2
    look_y = bounding_box[1]
    final_point_x = look_x - round(bounding_box[2] * math.sin(x_angle * math.pi / 180))
    final_point_y = look_y - round(bounding_box[2] * math.sin(y_angle * math.pi / 180))
    cv2.line(frame, (look_x, look_y), (final_point_x, final_point_y), color, 3)
    size = round((bounding_box[2] + bounding_box[3] / 2) / 8)
    cv2.circle(frame, (final_point_x, final_point_y), size, color, 2)

detector = MTCNN()
@jit
def detect(frame):
    return detector.detect_faces(frame)
        
        
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("output.mp4")

file_name = "save/log.txt"
file = open(file_name, 'w')
file.close()

while(True):
    file = open(file_name, 'a') 
    start = timer()
    ret, frame = cap.read()
    result = detect(frame)
    stats_log.append(len(result))
    look_log.append(0)
    time_log.append(global_time / 1000)
#    for i in range(len(bound)):
#        if (active_bound[i][0]):
#            cv2.rectangle(frame, (ex_bound[i][0], ex_bound[i][1]), (ex_bound[i][2], ex_bound[i][3]), (0, 155, 255), 1)
#            cv2.putText(frame, str(i), (ex_bound[i][0] + 2, ex_bound[i][1] + 14), cv2.FONT_HERSHEY_PLAIN,
#                1, (255, 155, 0), 1)
    
    
    for i in result:
        bounding_box = i['box']
        keypoints = i['keypoints']     
            
        #   Выбор нужного объекта слежения
        t = False
        uu = -1
        for j in range(len(ex_bound)):
            if (cam_math.in_range(ex_bound[j], bound[j], i['box']) and (active_bound[j][0] or (active_bound[j][1] == 'dead_by_timeout'))):
                if (not active_bound[j][0]):
                    print(str(j) + ' is reccurected')
                    active_bound[j] = [True, 'reccurected']
                if (not t):
                    t, uu = True, j
                else:
                    another_count[j] += 1
                    another_numb[j] = uu
                    

        if (t):
            count_bound[uu] = forget_rate        
            bound[uu] = i['box']  
            last_active[uu] = cycle_numb
            screen_time[uu] += 1
            time_last_seen[uu] = global_time
            check_minmaxes(bounding_box, bound_size_minmax[uu], bound_coord_minmax[uu])
        if (not t):
            bound.append(i['box'])
            ex_bound.append([])
            count_bound.append(forget_rate)
            active_bound.append([True, 'alive'])
            looking_time.append(1)
            first_active.append(cycle_numb)
            last_active.append(cycle_numb)
            active_time.append(0)
            screen_time.append(1)
            uu = len(ex_bound) - 1
            time_first_seen.append(global_time)
            time_last_seen.append(global_time)
            another_count.append(0)
            another_numb.append(-1)
            bound_size_minmax.append([[bounding_box[2], bounding_box[2]], [bounding_box[3], bounding_box[3]]])
            bound_coord_minmax.append([[bounding_box[0], bounding_box[0] + bounding_box[2]], 
                [bounding_box[1], bounding_box[1] + bounding_box[3]]])
        
        #   Создание области ожидания из текущего bounding_box
        a_ex_y = round(bounding_box[3] * ((ec - 1) / 2))
        a_ex_x = round(bounding_box[2] * ((ec - 1) / 2))
        a_x = (bound_coord_minmax[uu][0][0] + bounding_box[0]) // 2
        a_y = (bound_coord_minmax[uu][1][0] + bounding_box[1]) // 2
        a_xmax = (bound_coord_minmax[uu][0][1] + bounding_box[0] + bounding_box[2]) // 2
        a_ymax = (bound_coord_minmax[uu][1][1] + bounding_box[1] + bounding_box[2]) // 2
        a_sxmax = (bound_size_minmax[uu][0][1] + bounding_box[2]) // 2
        a_symax = (bound_size_minmax[uu][1][1] + bounding_box[3]) // 2
        a_sx = (bound_size_minmax[uu][0][0] + bounding_box[2]) // 2
        a_sy = (bound_size_minmax[uu][1][0] + bounding_box[3]) // 2
        ex = [a_x - a_ex_x, 
            a_y - a_ex_y,
            a_xmax + a_ex_x,
            a_ymax + a_ex_y,
            round(a_sxmax * size_ec), round(a_symax * size_ec), 
            round(a_sx / size_ec), round(a_sy / size_ec)]
        ex_bound[uu] = ex      
        
        #   Цвет bounding_box
        if cam_math.is_looking_on_screen(keypoints):
            color = (0, 255, 0)
            looking_time[uu] += 1
            look_log[len(look_log) - 1] += 1
        else:
            color = (0, 0, 255)
            
        draw_basics(frame, bounding_box, uu, keypoints, color)
            
        if (not t):
            cv2.imwrite("save/persons2/" + str(uu) + "_person.jpg", frame)
            cv2.imwrite("save/persons/" + str(uu) + "_person.jpg", frame)

        
    #"Память" программы, если в течении 500 циклов человек не появится в области ожидания, будет считаться, что он ушел
    for o in range(len(count_bound)):
        if (active_bound[o][0]):
            count_bound[o] -= 1
            active_time[o] += 1
            if (count_bound[o] <= 0):
                print(str(o) + ' is dead by timeout\n')
                file.write(str(global_time) + 'ms: ' + str(o) + ' is dead by timeout\n')
                active_bound[o] = [False, 'dead_by_timeout']
            if ((active_time[o] > 0) and (active_time[o] >= 50) and (screen_time[o] < 10)):
                print(str(o) + ' is ghost')
                file.write(str(global_time) + 'ms: ' + str(o) + ' is ghost\n')
                active_bound[o] = [False, 'ghost']
                os.remove("save/persons2/" + str(o) + "_person.jpg")
            
    #Согласование данных
    if (cycle_numb % 10 == 0):
        for i in range(len(another_count)):
            if (another_count[i] >= 6 and another_numb[i] >= 0):
                file.write(str(global_time) + 'ms: ' + 'set ' + str(i) + ' as ' + str(another_numb[i]) + '\n')
                print('set ' + str(i) + ' as ' + str(another_numb[i]))
                n = another_numb[i]
                active_bound[i] = [False, 'unity']      # Выключение i-того лица
                active_bound[n] = [True, 'alive']
                bound[n] = bound[i]                     # Согласование областей ожидания
                ex_bound[n] = ex_bound[i]
                count_bound[n] = forget_rate            # Обновление счетчика
                looking_time[n] += looking_time[i]      # Добавление looking_time
                screen_time[n] += screen_time[i]
                last_active[n] = last_active[i]
                active_time[n] = last_active[n] - first_active[n]
                unity_minmaxes(bound_coord_minmax[n], bound_coord_minmax[i],
                    bound_size_minmax[n], bound_size_minmax[i])
                os.remove("save/persons2/" + str(i) + "_person.jpg")
                if (time_last_seen[n] < time_last_seen[i]):
                    time_last_seen[n] = time_last_seen[i]
                
                
            another_count[i] = 0
            another_numb[i] = -1
    
    file.close()
    #   Вывод текущего кадра
    cv2.imshow('frame', frame)
    m = cv2.waitKey(100) & 0xFF
    if m == ord('q'):
        break
    elif m == ord('s'):
        cam_stats.save_stats(frame, time_last_seen, time_first_seen, screen_time, active_time, looking_time, bound, frame_list, active_bound)
    elif m == ord('g'):
        cam_stats.print_graph(time_log, stats_log, look_log)
    duration = timer() - start
    global_time += round(duration * 1000)
    cycle_numb += 1
        
cap.release()
cv2.destroyAllWindows()