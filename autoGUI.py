import gc
import pyautogui as pg
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import fingers
import handsign

# 0,0       X increases -->
# +---------------------------+
# |                           | Y increases
# |                           |     |
# |     1366 x 768 screen     |     |
# |                           |     V
# |                           |
# |                           |
# +---------------------------+ 1365, 767

# 0R - right click on/off
# 1R -
# 2R -
# 3R -
# 4R - left
# 5R - right
#
# 0L - doubleClick
# 1L - drag flag on/off --not working
# 2L - scroll flag --not working
# 3L - keyboard flag --not working
# 4L - up
# 5L - down
#?exit

finger_model = load_model("fingerCNNgood.h5")

hand_model = load_model("handsign.h5")

width, height = pg.size()

sign_dict = {
    0:'a',
    1:'b',
    2:'c',
    3:'d',
    4:'e',
    5:'f',
    6:'g',
    7:'h',
    8:'i',
    #j is movement
    9:'k',
    10:'l',
    11:'m',
    12:'n',
    13:'o',
    14:'p',
    15:'q',
    16:'r',
    17:'s',
    18:'t',
    19:'u',
    20:'v',
    21:'w',
    22:'x',
    23:'y'
    #z is movement
}

def rightClick():
    pg.mouseDown(button='right')
    pg.mouseUp(button='right')

def moveUp(drag=False, right=False, scroll=False):
    global width
    distance = width/13.5 if not right else width/45.5
    if scroll:
        scroll_width = width/4
        pg.scroll(scroll_width)
    elif drag:
        pg.dragRel(0, -distance, duration=0.2, button='left')
    else:
        pg.moveRel(0, -distance)

def moveDown(drag=False, right=False, scroll=False):
    global width
    distance = width/13.5 if not right else width/45.5 #100, 30
    if scroll:
        scroll_width = width/4
        pg.scroll(-scroll_width) #350
    elif drag:
        pg.dragRel(0, distance, duration=0.2, button='left')
    else:
        pg.moveRel(0, distance)

def moveLeft(drag=False, right=False, scroll=False):
    global height
    distance = height/10 if not right else height/25.5 #75, 30
    if scroll:
        scroll_height = height/2
        pg.hscroll(-scroll_height)
    elif drag:
        pg.dragRel(-distance, 0, duration=0.2, button='left')
    else:
        pg.moveRel(-distance, 0)

def moveRight(drag=False, right=False, scroll=False):
    global height
    distance = height/10 if not right else height/25.5
    if scroll:
        scroll_height = height/2
        pg.hscroll(scroll_height)
    elif drag:
        pg.dragRel(distance, 0, duration=0.2, button='left')
    else:
        pg.moveRel(distance, 0)

def captureFingers():
    global finger_model
    success, img = cv2.VideoCapture(0).read()
    image = fingers.preprocess(img)
    #y_pred = np.argmax(finger_model.predict(image), axis=1)
    y_predlist = finger_model.predict(image)
    y_pred = np.argmax(y_predlist)
    print(y_predlist)
    print(y_pred)
    print(fingers.label_list[y_pred])
    #_ = gc.collect()
    return fingers.label_list[y_pred]

def captureHandSign():
    global hand_model
    success, img = cv2.VideoCapture(0).read()
    image = handsign.preprocess(img)
    #y_pred = np.argmax(hand_model.predict(image), axis=1)
    y_predlist = hand_model.predict(image)
    y_pred = np.argmax(y_predlist, axis=1)
    print(y_predlist)
    print(y_pred)
    return max(y_pred)

def run(start_photo):
    #drag if it is meant to be dragged
    DRAG_FLAG = False
    #right if right click was pressed
    RIGHT_FLAG = False
    #scroll if we want to scroll
    SCROLL_FLAG = False
    #keyboard if we want to type
    KEYBOARD_FLAG = False
    pg.moveTo(pg.locateCenterOnScreen(start_photo))
    while 1:
        prediction = None
        _ = gc.collect()
        #time.sleep(1)
        if KEYBOARD_FLAG:
            prediction = captureHandSign()
            if sign_dict[prediction] == "w":
                KEYBOARD_FLAG = False
                print("KEYBOARD_FLAG: On")
            else:
                pg.write(sign_dict[prediction])
        else:
            pass
            prediction = captureFingers()
            if prediction == "0R":
                RIGHT_FLAG = False if RIGHT_FLAG else True
                print("RIGHT CLICK: " + ("On" if RIGHT_FLAG else "Off"))
                if RIGHT_FLAG:
                    rightClick()
                else:
                    pg.click()
            elif prediction == "1R":
                pass
            elif prediction == "2R":
                pass
            elif prediction == "3R":
                pass
            elif prediction == "4R":
                moveLeft(DRAG_FLAG, RIGHT_FLAG, SCROLL_FLAG)
            elif prediction == "5R":
                moveRight(DRAG_FLAG, RIGHT_FLAG, SCROLL_FLAG)
            elif prediction == "0L":
                pg.doubleClick()
            elif prediction == "1L":
                #DRAG_FLAG = False if DRAG_FLAG else True
                print("DRAG: " + ("On" if DRAG_FLAG else "Off"))
            elif prediction == "2L":
                #SCROLL_FLAG = False if SCROLL_FLAG else True
                print("SCROLL: " + ("On" if SCROLL_FLAG else "Off"))
            elif prediction == "3L":
                #KEYBOARD_FLAG = False if KEYBOARD_FLAG else True
                print("KEYBOARD_FLAG: " + ("On" if KEYBOARD_FLAG else "Off"))
            elif prediction == "4L":
                moveDown(DRAG_FLAG, RIGHT_FLAG, SCROLL_FLAG)
            elif prediction == "5L":
                moveUp(DRAG_FLAG, RIGHT_FLAG, SCROLL_FLAG)