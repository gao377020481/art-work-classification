import matplotlib.pyplot as plt
name_list = ['Edgar', 'Mikhail', 'Picasso', 'Rembrandt','van Gogh']
num_list = [560, 133, 356, 215, 710]
rects=plt.bar(range(len(num_list)), num_list, color='rgby')
index=[0,1,2,3,4]
index=[float(c)+0.4 for c in index]
plt.ylim(top=750, bottom=0)
plt.xticks(index, name_list)
plt.ylabel("train set numbers")
for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center',va='bottom')
plt.show()
