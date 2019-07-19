 
from numba import jit
#Углы для проверки верного направления взгляда
x_angle_plus = 60
x_angle_minus = 10
y_angle_plus = 30
y_angle_minus = -10

#Проверка направления взгляда примерно в сторону экрана
@jit
def is_looking_on_screen(key):
    angle = get_angle(key)
    return ((angle[0] >= x_angle_minus) & (angle[0] <= x_angle_plus) &
        (angle[1] >= y_angle_minus) & (angle[1] <= y_angle_plus))
        


#Получение угла наклона головы: по осям oy и ox
@jit
def get_angle(key):
    p1, p2, p3, p4, p5 = key['left_eye'], key['right_eye'], key['nose'], key['mouth_left'], key['mouth_right']
    x1, x2, x3 = p1[0], p2[0], p3[0]
    y1, y2, y3, y4, y5 = p1[1], p2[1], p3[1], p4[1], p5[1]
    c1, c2, x = 0, 0, 0
    dx = x2 - x1
    dy = y2 - y1
    if (dy != 0):
        a1 = dy / dx
        b1 = -x1 * dy / dx + y1
        a2 = -dx / dy
        b2 = y3 + dx * x3 / dy
        x = (b2 - b1) / (a1 - a2)
    else:
        x = x3
    dxx = x2 - x
    x_angle = round(140 * ((x1 + x2) / 2 - x) / dx)
    y_angle = round(200 * (0.55 - (y3 - (y2 + y1) / 2) / ((y4 + y5) / 2 - (y2 + y1) / 2)))
    return [x_angle, y_angle]

#Проверка нахождения bounding_box в области ожидания   
@jit 
def in_range(extrapol, lastbox, box):
	x, y, x2, y2 = extrapol[0], extrapol[1], extrapol[2], extrapol[3]
	bx, by, bsx, bsy = box[0], box[1], box[2], box[3]
	if (x <= 0):
		x = 0
	if (y <= 0):
		y = 0
	if (bx <= 0):
		bx = 0
	if (by <= 0):
		by = 0
	return (x <= bx) and (y <= by) and (x2 >= bx + bsx) and (y2 >= by + bsy) and (bsx <= extrapol[4]) and (bsx >= extrapol[6]) and (bsy <= extrapol[5]) and (bsy >= extrapol[7])