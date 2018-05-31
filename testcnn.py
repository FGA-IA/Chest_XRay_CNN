import os

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from numpy import set_printoptions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.chdir(r'.')
set_printoptions(precision=4, suppress=True)
os.listdir('.')
rede = '/redes/chest_xray_cnn_0.64.h5'

my_model = load_model(filepath='./' + rede)
print(my_model.summary())

eval_idg = ImageDataGenerator(rescale=1. / 255)
eval_g = eval_idg.flow_from_directory(directory=r'data/val/',
                                     target_size=(100,100),
                                     class_mode='binary',
                                     batch_size=20,
                                     shuffle=False)
(eval_loss, eval_acc) = my_model.evaluate_generator(generator=eval_g, steps=1)
print('Evaluation Loss over never-before-seen images is: {:.4f}'.format(eval_loss))
print('Evaluation Accuracy over never-before-seen images is: {:4.2f}'.format(eval_acc*100))

pred_idg = eval_idg
pred_g = eval_g
pred = my_model.predict_generator(generator=pred_g, steps=1)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print(pred_g.class_indices, '\n')

print(len(pred_g.filenames))

for x in range(len(pred_g.filenames)):
    if (pred[x]) <= 0.5:
        print('Normal-');
    else:
        print('Pneumonia-')
    print(str(pred[x]), '\n')
    name = 'data/val/' + str(pred_g.filenames[x])
    print(name)
    img=mpimg.imread(name)
    imgplot = plt.imshow(img)
    plt.show()