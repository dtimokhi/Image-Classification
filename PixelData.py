from PIL import Image as im
import os

frowns = [file for file in os.listdir("60frowns") if file[-7:] != '(1).jpg']
smiles = [file for file in os.listdir("60smiles") if file[-7:] != '(1).jpg' and file != '.ipynb_checkpoints']

file = open('data.csv', 'w')
file.write(",".join(frowns) + "," + ",".join(smiles) + "\n")

for f in frowns:
    path = '60frowns/' + f
    image = im.open(path, 'r')
    pixels = list([str(x) for x in image.getdata()])
    if len(pixels) != 3600:
        print(len(pixels))
        print(f)
    file.write(",".join(pixels) + "," + "frown\n")
for s in smiles:
    path = '60smiles/' + s
    image = im.open(path, 'r')
    pixels = list([str(x) for x in image.getdata()])
    if len(pixels) != 3600:
        print(len(pixels))
        print(s)
    file.write(",".join(pixels) + "," + "smile\n")
file.close()
