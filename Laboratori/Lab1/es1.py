import functools



class Athlete:

    def __init__(self, name, surname, nationality, score_list):
        self.name = name
        self.surname = surname
        self.nationality = nationality
        self.score_list = score_list
        self.totalScore = -1

    def computeScore(self):
        sum = 0

        #new_list = list(self.score_list)

        new_list = sorted(self.score_list)    #ordino la lista
        new_list.pop(0)      #elimino il primo elemento
        new_list.pop()       #elimino l'ultimo elemento

        for val in new_list:
            sum = sum + val

        self.totalScore = sum

        return self.totalScore
    
    def __str__(self):
        return f'name = {self.name}\n surname = {self.surname}\n nationality = {self.nationality}\n score_list = {self.score_list}\n totalScore = {self.totalScore}'

    def __repr__(self):
        str = f'Athlete(\'{self.name}\', {self.surname}, \'{self.nationality}\', \'{self.score_list}\', \'{self.totalScore}\')'
        return str
    
    def printTotalScore(self):
        print(f'{self.name} {self.surname} - Score: {self.totalScore}')
    

def readFile(file):
    
    athletes = []
    name = ''
    surname = ''
    nationality = ''

    with open(file, 'r') as fin:
        for line in fin:
            scores = []
            fields = line.split()

            name = fields[0]
            surname = fields[1]
            nationality = fields[2]

            for i in range(3, 8):
                scores.append(float(fields[i]))

            athlete = Athlete(name, surname, nationality, scores)

            athletes.append(athlete) 


        return athletes   
    
def sortingAthletesByTotScore(a, b):

    return -1 if a.totalScore > b.totalScore else 1  # Descending order


athletes = readFile('ex1_data.txt')

for athlete in athletes:
    totScore = athlete.computeScore()

print('\nCOMPUTED TOTAL SCORES:\n')
print(*athletes, sep = '\n')

athletes = sorted(athletes, key = functools.cmp_to_key(sortingAthletesByTotScore))
selected_athletes = athletes[0:3]

print('\n--------BEST 3 ATHLETES---------\n')
for i in range(3):
    print(f'{i}: ', end = '')
    selected_athletes[i].printTotalScore()

dictOfCountries = {}

for athlete in athletes:
    country = athlete.nationality

    try:
        dictOfCountries[country] += athlete.totalScore
    except KeyError:
        dictOfCountries[country] = athlete.totalScore


sortedDictCountries = sorted(dictOfCountries)

bestCountryName = list(dictOfCountries.keys())[0]
bestCountryScore = list(dictOfCountries.values())[0]

print('\n--------BEST COUNTRY---------\n')
print(f'{bestCountryName} - Total score: {bestCountryScore}')