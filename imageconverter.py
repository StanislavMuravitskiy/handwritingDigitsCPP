from PIL import Image

import numpy as np

img = Image.open('img.jpg').convert('RGB')
#img.show()
img = img.resize((28,28))
    # конвертируем rgb в grayscale
img = img.convert('L')
img = np.array(img)

img.reshape((28, 28))




lst = list(img)

for i in range(28):
    for j in range(28):
        if lst[i][j] < 10:
            lst[i][j] = 0




file = open("img.txt", 'w')
for el in lst:
    print(*el, file=file)
file.close()



