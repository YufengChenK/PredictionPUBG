import pandas as pd

trainData = pd.read_csv("train.csv")

destroys_ = trainData.ix[:, "vehicleDestroys"]
# print(destroys_)
# print(type(destroys_))
count0 = 0
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
count6 = 0
count7 = 0
count8 = 0
count9 = 0
print("*******")
for x in destroys_:
    if x == 0:
        count0 += 1
    if x == 1:
        count1 += 1
    if x == 2:
        count2 += 1
    if x == 3:
        count3 += 1
    if x == 4:
        count4 += 1
    if x == 5:
        count5 += 1
    if x == 6:
        count6 += 1
    if x == 7:
        count7 += 1
    if x == 8:
        count8 += 1
    if x == 9:
        count9 += 1

print("count0", count0)
print("count1", count1)
print("count2", count2)
print("count3", count3)
print("count4", count4)
print("count5", count5)
print("count6", count6)
print("count7", count7)
print("count8", count8)
print("count9", count9)
