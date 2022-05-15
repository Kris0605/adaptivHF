import numpy as np
import json

from sympy import false, true


class RemotePlayerStrategy:
    def __init__(self, **kwargs):
        self.nextAction = "0"
        self.sendData = kwargs["sender"]
        self.getData = kwargs["getter"]
        self.name = kwargs["name"]

    def setObservations(self, ownObject, fieldDict):
        self.nextAction = "0"
        self.sendData(json.dumps({"type":"gameData", "payload":fieldDict}), ownObject.name)

    def getNextAction(self):
        newaction = self.getData(self.name)
        if newaction is None:
            return self.nextAction
        else:
            return newaction

    def reset(self):
        self.nextAction = "0"
        data = "something"
        while data is not None:
            data = self.getData(self.name)

class DummyStrategy:
    def __init__(self, **kwargs):
        self.nextAction = "0"

    def setObservations(self, ownObject, fieldDict):
        ownObject.active=False

    def getNextAction(self):
        return "0"

    def reset(self):
        pass

class RandBotStrategy:
    def __init__(self, **kwargs):
        self.nextAction = 0

    def setObservations(self, ownObject, fieldDict):
        pass

    def getNextAction(self):
        actdict = {0: "0", 1: "+", 2: "-"}
        r = np.random.randint(0, 3, 2)
        action = ""
        for act in r:
            action += actdict[act]

        return action

    def reset(self):
        self.nextAction = "0"

class NaiveStrategy:
    def __init__(self, **kwargs):
        self.nextAction = "0"
        self.oldpos = None
        self.oldcounter = 0

    def getRandomAction(self):
        actdict = {0: "0", 1: "+", 2: "-"}
        r = np.random.randint(0, 3, 2)
        action = ""
        for act in r:
            action += actdict[act]

        return action

    def setObservations(self, ownObject, fieldDict):
        if self.oldpos is not None:
            if tuple(self.oldpos) == tuple(ownObject.pos):
                self.oldcounter += 1

        self.oldpos = ownObject.pos.copy()

        values = np.array([field["value"] for field in fieldDict["vision"]])
        values[values > 3] = 0
        values[values < 0] = 0
        if np.max(values) == 0 or self.oldcounter >= 3:
            self.nextAction = self.getRandomAction()
            self.oldcounter = 0
        else:
            idx = np.argmax(values)
            actstring = ""
            for i in range(2):
                if fieldDict["vision"][idx]["relative_coord"][i] == 0:
                    actstring += "0"
                elif fieldDict["vision"][idx]["relative_coord"][i] > 0:
                    actstring += "+"
                elif fieldDict["vision"][idx]["relative_coord"][i] < 0:
                    actstring += "-"

            self.nextAction = actstring

    def getNextAction(self):
        return self.nextAction

    def reset(self):
        self.nextAction = "0"

class NaiveHunterStrategy:
    def __init__(self, **kwargs):
        self.nextAction = "0"
        self.oldpos = None
        self.oldcounter = 0

    def getRandomAction(self):
        actdict = {0: "0", 1: "+", 2: "-"}
        r = np.random.randint(0, 3, 2)
        action = ""
        for act in r:
            action += actdict[act]

        return action

    def setObservations(self, ownObject, fieldDict):
        if self.oldpos is not None:
            if tuple(self.oldpos) == tuple(ownObject.pos):
                self.oldcounter += 1
            else:
                self.oldcounter = 0
        if ownObject.active:
            self.oldpos = ownObject.pos.copy()

        vals = []
        for field in fieldDict["vision"]:
            if field["player"] is not None:
                if tuple(field["relative_coord"]) == (0, 0):
                    if 0 < field["value"] <= 3:
                        vals.append(field["value"])
                    elif field["value"] == 9:
                        vals.append(-1)
                    else:
                        vals.append(0)
                elif field["player"]["size"] * 1.1 < ownObject.size:
                    vals.append(field["player"]["size"])
                else:
                    vals.append(-1)
            else:
                if 0 < field["value"] <= 3:
                    vals.append(field["value"])
                elif field["value"] == 9:
                    vals.append(-1)
                else:
                    vals.append(0)

        values = np.array(vals)
        # print(values, fieldDict["vision"][np.argmax(values)]["relative_coord"], values.max())
        if np.max(values) <= 0 or self.oldcounter >= 3:
            self.nextAction = self.getRandomAction()
            self.oldcounter = 0
        else:
            idx = np.argmax(values)
            actstring = ""
            for i in range(2):
                if fieldDict["vision"][idx]["relative_coord"][i] == 0:
                    actstring += "0"
                elif fieldDict["vision"][idx]["relative_coord"][i] > 0:
                    actstring += "+"
                elif fieldDict["vision"][idx]["relative_coord"][i] < 0:
                    actstring += "-"

            self.nextAction = actstring

    def getNextAction(self):
        return self.nextAction

    def reset(self):
        self.nextAction = "0"

class AdvancedPlayerStrategy:
    def __init__(self, **kwargs):
        self.nextAction = "0"
        self.oldpos = None
        self.oldcounter = 0
        self.map = np.zeros((40,40))
        self.currentX=None
        self.currentY=None

    def getRandomAction(self):
        actdict = {0: "0", 1: "+", 2: "-"}
        r = np.random.randint(0, 3, 2)
        action = ""
        for act in r:
            action += actdict[act]

        return action
    
    def GoCenter(self):
        action = "00"
        return action

    def setObservations(self, ownObject, fieldDict):
        if self.oldpos is not None:
            if tuple(self.oldpos) == tuple(ownObject.pos):
                self.oldcounter += 1
            else:
                self.oldcounter = 0
        if ownObject.active:
            self.oldpos = ownObject.pos.copy()
            self.currentX=ownObject.pos[0]
            self.currentY=ownObject.pos[1]

        vision = np.zeros((11,11))
        for field in fieldDict["vision"]:
            #print(field["relative_coord"][0], end ="/")
            if field["player"] is not None:
                if tuple(field["relative_coord"]) == (0, 0):
                    if 0 < field["value"] <= 3:
                        vision[field["relative_coord"][1]+5][field["relative_coord"][0]+5]=field["value"]
                    elif field["value"] == 9:
                        vision[field["relative_coord"][1]+5][field["relative_coord"][0]+5]=-1
                    else:
                        vision[field["relative_coord"][1]+5][field["relative_coord"][0]+5]=0
                elif field["player"]["size"] * 1.1 < ownObject.size:
                    vision[field["relative_coord"][1]+5][field["relative_coord"][0]+5]=field["player"]["size"]
                else:
                    vision[field["relative_coord"][1]+5][field["relative_coord"][0]+5]=-1
            else:
                if 0 < field["value"] <= 3:
                    vision[field["relative_coord"][1]+5][field["relative_coord"][0]+5]=field["value"]
                elif field["value"] == 9:
                    vision[field["relative_coord"][1]+5][field["relative_coord"][0]+5]=-1
                else:
                    vision[field["relative_coord"][1]+5][field["relative_coord"][0]+5]=0

        if(self.currentX<5):
            if(self.currentY<5):
                self.map[:self.currentY+6, :self.currentX+6]=vision[(5-self.currentY):, (5-self.currentX):]
            elif(self.currentY>34):
                self.map[self.currentY-5:, :self.currentX+6]=vision[:5+(40-self.currentY), (5-self.currentX):]
            else:
                self.map[self.currentY-5:self.currentY+6, :self.currentX+6]=vision[:, (5-self.currentX):]
        elif(self.currentX>34):
            if(self.currentY<5):
                self.map[:self.currentY+6, self.currentX-5:]=vision[(5-self.currentY):, :5+(40-self.currentX)]
            elif(self.currentY>34):
                self.map[self.currentY-5:, self.currentX-5:]=vision[:5+(40-self.currentY), :5+(40-self.currentX)]
            else:
                self.map[self.currentY-5:self.currentY+6, self.currentX-5:]=vision[:, :5+(40-self.currentX)]
        else:
            if(self.currentY<5):
                self.map[:self.currentY+6, self.currentX-5:self.currentX+6]=vision[(5-self.currentY):,:]
            elif(self.currentY>34):
                self.map[self.currentY-5:, self.currentX-5:self.currentX+6]=vision[:5+(40-self.currentY),:]
            else:
                self.map[self.currentY-5:self.currentY+6, self.currentX-5:self.currentX+6]=vision
        print(self.map)

        

        # print(values, fieldDict["vision"][np.argmax(values)]["relative_coord"], values.max())
        if np.max(vision) <= 0 or self.oldcounter >= 3:
            self.nextAction = self.GoCenter()
            self.oldcounter = 0
        else:
            print(vision)
            maxIndex= np.argmax(vision)
            maxIndexX = maxIndex % 11
            maxIndexY = int(maxIndex/11)
            relativeCoord=[maxIndexX-5,maxIndexY-5]
            
            actstring = ""
            for i in range(2):
                if relativeCoord[i] == 0:
                    actstring += "0"
                elif relativeCoord[i] > 0:
                    actstring += "+"
                elif relativeCoord[i] < 0:
                    actstring += "-"
            self.nextAction = actstring

    def getNextAction(self):
        return self.nextAction

    def reset(self):
        self.nextAction = "0"

class Player:
    strategies = {"randombot": RandBotStrategy, "naivebot": NaiveStrategy, "naivehunterbot": NaiveHunterStrategy,
                  "remoteplayer": RemotePlayerStrategy, "dummy":DummyStrategy, "advanced":AdvancedPlayerStrategy}

    def __init__(self, name, playerType, startingSize, **kwargs):
        self.name = name
        self.playerType = playerType
        self.pos = np.zeros((2,))
        self.size = startingSize
        kwargs["name"] = name
        self.strategy = Player.strategies[playerType](**kwargs)
        self.active = True

    def die(self):
        self.active = False
        print(self.name + " died!")

    def reset(self):
        self.active = True

