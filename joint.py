from PIL import Image
import cv2
arr = ['B:\\origin.png', 'B:\\result_5.png', 'B:\\result_15.png', 'B:\\result_25.png']
toImage = Image.new('RGBA',(1000,1000))
for i in range(4):
    fromImge = Image.open(arr[i])
    # loc = ((i % 2) * 200, (int(i/2) * 200))
    loc = ((int(i/2) * 500), (i % 2) * 500)
    print(loc)
    toImage.paste(fromImge, loc)

toImage.save('B:\\go11.png')

img = cv2.imread('B:\\go11.png')
print(type(img))
cv2.imshow(img)
cv2.waitKey(0)
