from os import system
import glob

N = 10000
for file in glob.iglob("OCR_dataset_2/*.jpg"):
    system("cp " + file + " OCR_dataset_3/")
    N = N - 1
    print(N)
    if N == 0:
        break