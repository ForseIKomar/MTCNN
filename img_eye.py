#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import cam_math
import datetime
import math

ec = 1.2                # Соотношение размеров области ожидания к bounding_box
forget_rate = 50       # Количество циклов, необходимых для забывания объекта

bound = []              # Массив bounding_box
ex_bound = []           # Массив областей ожидания
active_bound = []       # Буленовский массив, отражает активные/выключенные bounding_box
count_bound = []        # Массив чисел, для забывания объекта

active_time = []
looking_time = []
screen_time = []

def print_stats():
    for i in range(len(bound)):
        if (active_bound[i]):
            print(str(i) + ": время наблюдения = " + str(round(active_time[i] * 1.23) / 10) +
                " секунд, процент внимания = " + str(round(1000 * (looking_time[i] / active_time[i])) / 10) + 
                ", процент нахождения на экране = " + str(round(1000 * (screen_time[i] / active_time[i])) / 10))

def save_stats(img):
    file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.txt")
    file = open(file_name, 'w')
    for i in range(len(bound)):
        if (active_bound[i]):
            file.write(str(i) + ": время наблюдения = " + str(round(active_time[i] * 1.23) / 10) +
                " секунд, процент внимания = " + str(round(1000 * (looking_time[i] / active_time[i])) / 10) + 
                ", процент нахождения на экране = " + str(round(1000 * (screen_time[i] / active_time[i])) / 10) + "\n")
    file.close()
    cv2.imwrite(datetime.datetime.now().strftime("%Y%m%d_%H%M%S.jpg"), img)
    print(file_name + ' saving done')
                

detector = MTCNN()

def main():
    frame = cv2.imread("1.jpg")
    result = detector.detect_faces(frame)
    for i in range(len(result)):
        bounding_box = result[i]['box']
        keypoints = result[i]['keypoints']     
        #   Выбор нужного объекта слежения
        t = False
        uu = 0
        for j in range(len(ex_bound)):
            if (cam_math.in_range(ex_bound[j], bounding_box) & active_bound[j]):
                t, uu = True, j
       
        #   Создание области ожидания из текущего bounding_box
        ex = [bounding_box[0] - round(bounding_box[2] * ((ec - 1) / 2)), 
			bounding_box[1] - round(bounding_box[3] * ((ec - 1) / 2)),
			bounding_box[0] + round(bounding_box[2] * ((ec + 1) / 2)),
			bounding_box[1] + round(bounding_box[3] * ((ec + 1) / 2))]
        if (t):
            count_bound[uu] = forget_rate        
            bound[uu] = bounding_box
            ex_bound[uu] = ex
        if (not t):
            bound.append(bounding_box)
            ex_bound.append(ex)
            active_bound.append(True)
            count_bound.append(forget_rate)
            looking_time.append(1)
            active_time.append(0)
            screen_time.append(0)
            uu = len(ex_bound) - 1
        
        screen_time[uu] += 1
        
        
        angle = cam_math.get_angle(keypoints)
        x_angle = angle[0]
        y_angle = angle[1]
         
#        cv2.putText(frame, str([x_angle, y_angle]), (10, 100), cv2.FONT_HERSHEY_PLAIN,
#            1, (0, 155, 255), 2)
        
#        cv2.putText(frame, str([looking_time[uu], screen_time[uu]]) + " "
#        + str([screen_time[uu], active_time[uu]]), (10, 200 + uu * 20), cv2.FONT_HERSHEY_PLAIN,
#            1, (0, 155, 255), 2)
        
        #   Цвет bounding_box
        if cam_math.is_looking_on_screen(keypoints):
            color = (0, 255, 0)
            looking_time[uu] += 1
        else:
            color = (0, 0, 255)
            
        #Прямоугольник лица человека
        cv2.rectangle(frame,
            (bounding_box[0], bounding_box[1]),
            (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
            color,
            2)

        cv2.putText(frame, str(uu), (bounding_box[0] + 2, bounding_box[1] + 28), cv2.FONT_HERSHEY_PLAIN,
            2, (255, 255, 255), 1)
        
#        cv2.circle(frame,(keypoints['left_eye']), 1, (0,155,255), 2)
#        cv2.circle(frame,(keypoints['right_eye']), 1, (0,155,255), 2)
#        cv2.circle(frame,(keypoints['nose']), 1, (0,155,255), 2)
#        cv2.circle(frame,(keypoints['mouth_left']), 1, (0,155,255), 2)
#        cv2.circle(frame,(keypoints['mouth_right']), 1, (0,155,255), 2)

        
        #Прямоугольник ожидания
        cv2.rectangle(frame,
            (bounding_box[0] - round(bounding_box[2] * ((ec - 1) / 2)), 
            bounding_box[1] - round(bounding_box[3] * ((ec - 1) / 2))),
            (bounding_box[0]+round(bounding_box[2] * ((ec + 1) / 2)),
            bounding_box[1] + round(bounding_box[3] * ((ec + 1) / 2))),
            (0,155,255),
            2)
            
        #Указатель направления взгляда
        look_x = bounding_box[0] + bounding_box[2] // 2
        look_y = bounding_box[1]
        final_point_x = look_x - round(bounding_box[2] * math.sin(x_angle * math.pi / 180))
        final_point_y = look_y - round(bounding_box[2] * math.sin(y_angle * math.pi / 180))
        cv2.line(frame, (look_x, look_y), (final_point_x, final_point_y), color, 3)
        cv2.circle(frame, (final_point_x, final_point_y), 10, color, 4)
            
    #"Память" программы, если в течении 400 циклов человек не появится в области ожидания, будет считаться, что он ушел
    for o in range(len(count_bound)):
        count_bound[o] -= 1
        active_time[o] += 1
#        cv2.circle(frame, (bound[o][0] + bound[o][2] // 2, bound[o][1] + bound[o][3] // 2), 20, color, 4)
        if (count_bound[o] <= 0):
            active_bound[o] = False
    
    #   Вывод текущего кадра
    cv2.imwrite('1_ex.jpg', frame)
    m = cv2.waitKey(30) & 0xFF
        
main()
        
cv2.destroyAllWindows()