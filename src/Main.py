from Engine import *
import time

engine = AdaptIOEngine("./gui/maps/base_field.txt",5,1.2,{"Teszt":"naivebot","Teszt1":"randombot","Teszt2":"randombot","Teszt3":"randombot"},5,"static")

stime = time.time()
for i in range(300):
    engine.tick()

print(time.time()-stime)