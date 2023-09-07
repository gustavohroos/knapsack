import requests 
import os

base = "https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/"

for i in range(1, 9):
    if not os.path.exists("data/p0" + str(i) + "/"):
        os.makedirs("data/p0" + str(i))

    for file in ['_c.txt', '_p.txt', '_w.txt', '_s.txt']:
        url = base + "p0" + str(i) + file
        r = requests.get(url, allow_redirects=True)
        print(url)
        print(r.headers.get('content-type'))
        open('data/p0' + str(i) + '/p0' + str(i) + file, 'wb').write(r.content)
    



