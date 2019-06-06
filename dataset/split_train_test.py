import random

img_num=590
test_num=50
img_ids=list(range(1,img_num+1))
test_ids=random.sample(img_ids,test_num)

train_ids=[]
for i in range(1,img_num+1):
    if(i not in test_ids):
        train_ids.append(i)

f=open('test_ids.txt','w')
for i in range(len(test_ids)):
    f.write('%05d\n'%test_ids[i])
f.close()

f=open('train_ids.txt','w')
for i in range(len(train_ids)):
    f.write('%05d\n'%train_ids[i])
f.close()
