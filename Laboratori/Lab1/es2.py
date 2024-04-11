
import sys

class BusInfo:
    def __init__(self, busID, lineID, x, y, time):
        self.busID = busID
        self.lineID = lineID
        self.x = int(x)
        self.y = int(y)
        self.time = int(time)



if __name__ == '__main__':

    
    entryVett = []

    with open(sys.argv[1]) as fin:
        
        for line in fin:
            busID, lineID, x, y, time = line.split()  #separo dove trovo spazi bianchi
            entryVett.append(BusInfo(busID, lineID, x, y, time))

    if(sys.argv[2] == '-b'):    #il prossimo parametro e' un busID

        totalDistanceBus = 0
        old_x = 0
        old_y = 0
        for entry in entryVett:
            
            if entry.busID == sys.argv[3]:  #ho trovato il bus di interesse

                if(old_x != 0 and old_y != 0):
                    diff_x = abs(entry.x - old_x)
                    diff_y = abs(entry.y - old_y)
                    distance = (diff_x**2 + diff_y**2)**(1/2)

                    totalDistanceBus += distance

                old_x = entry.x
                old_y = entry.y

        print(f'{sys.argv[3]} - Total Distance = {totalDistanceBus}')

    elif(sys.argv[2] == '-l'):  #il prossimo parametro e' una linea

        speedSum = 0
        numberOfEntries = 0
        old_x = 0
        old_y = 0
        old_time = 0

        for entry in entryVett:

            if entry.lineID == sys.argv[3]: #ho trovato la linea di interesse

                if(old_x != 0 and old_y != 0 and old_time != 0):
                    diff_x = abs(entry.x - old_x)
                    diff_y = abs(entry.y - old_y)
                    distance = (diff_x**2 + diff_y**2)**(1/2)
                    time = entry.time - old_time

                    speed = distance / time

                    speedSum += speed
                    numberOfEntries = numberOfEntries + 1
                
                old_x = entry.x
                old_y = entry.y
                old_time = entry.time

        avgSpeed = speedSum / numberOfEntries
        print(f'{sys.argv[3]} - Avg Speed = {avgSpeed}')





