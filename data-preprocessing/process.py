import numpy as np
import cv2
import pickle
from functions import concatenate, normalize, resize, bgr_to_rgb, image_from_array, flatten, to_float

### FOLDER LABELS ###
sensornames = ['color', 'depthcolormap', 'icub_left', 'icub_right']
toolnames = ['hook', 'ruler', 'spatula', 'sshot']
actions = ['left_to_right', 'pull', 'push', 'right_to_left']
objectnames = ['0_woodenCube', '1_pearToy', '2_yogurtYellowbottle', '3_cowToy', '4_tennisBallYellowGreen',
               '5_blackCoinbag', '6_lemonSodaCan', '7_peperoneGreenToy', '8_boxEgg', '9_pumpkinToy',
               '10_tomatoCan', '11_boxMilk', '12_containerNuts', '13_cornCob', '14_yellowFruitToy',
               '15_bottleNailPolisher', '16_boxRealSense', '17_clampOrange', '18_greenRectangleToy', '19_ketchupToy']

def preprocess_image(path, sensor, j, width, height):
    if sensor == 'icub_right' or sensor == 'icub_left':
        init = cv2.imread(path + 'init_color_' + sensor + '_' + str(j) + '.png')
        effect = cv2.imread(path + 'effect_color_' + sensor + '_' + str(j) + '.png')
    else:
        init = cv2.imread(path + 'init_' + sensor + '_' + str(j) + '.png')
        effect = cv2.imread(path + 'effect_' + sensor + '_' + str(j) + '.png')

    # Pre-processing steps
    init = resize(init, width, height)
    effect = resize(effect, width, height)
    init = bgr_to_rgb(init)
    effect = bgr_to_rgb(effect)
    init = normalize(init)
    effect = normalize(effect)
    image = concatenate(init, effect)
    image = image.reshape(1, *image.shape)
    return image

def save_data_to_disk(data, path):
    pickle.dump(data, open(path, 'wb'))

def main():
    width, height = 256, 192

    # This is size of the processed image not the raw
    size_image = (height, width*2, 3)

    # Training
    training_color = np.zeros((1, *size_image), dtype=np.float32)
    training_icub_left = np.zeros((1, *size_image), dtype=np.float32)
    training_icub_right = np.zeros((1, *size_image), dtype=np.float32)

    # Validation
    validation_color = np.zeros((1, *size_image), dtype=np.float32)
    validation_icub_left = np.zeros((1, *size_image), dtype=np.float32)
    validation_icub_right = np.zeros((1, *size_image), dtype=np.float32)

    # Test
    testing_color = np.zeros((1, *size_image), dtype=np.float32)
    testing_icub_left = np.zeros((1, *size_image), dtype=np.float32)
    testing_icub_right = np.zeros((1, *size_image), dtype=np.float32)

    # Training depth images
    training_depth = np.zeros((1, *size_image), dtype=np.float32)

    # Validation depth images
    validation_depth = np.zeros((1, *size_image), dtype=np.float32)

    # Test depth images
    testing_depth = np.zeros((1, *size_image), dtype=np.float32)

    # Checking shapes
    print("Training shapes:", training_color.shape, training_icub_left.shape, training_icub_right.shape, training_depth.shape)
    print("Validation shapes:", validation_color.shape, validation_icub_left.shape, validation_icub_right.shape, validation_depth.shape)
    print("Testing shapes:", testing_color.shape, testing_icub_left.shape, testing_icub_right.shape, testing_depth.shape)

    y_training, y_validation, y_testing = [], [], []

    for a in range(len(objectnames)):
        objectname = objectnames[a]
        for y in range(len(toolnames)):
            toolname = toolnames[y]
            for x in range(len(actions)):
                action = actions[x]
                label_action = actions.index(action)
                label_tool = toolnames.index(toolname)

                # Split into sets: 60% Training, 20% Validation, 20% Testing
                ids = np.random.choice(np.arange(10), 10, replace=False)
                training_ids, validation_ids, testing_ids = ids[0:6], ids[6:8], ids[8:10]
                
                for i in range(len(sensornames)):
                    sensor = sensornames[i]
                    path = 'C:/Users/Frank/OneDrive/Bureaublad/action_recognition_dataset/' + objectname + '/' + toolname + '/' + action + '/' + sensor + '/'

                    # Loop through the number of repeats
                    for j in range(10):
                        image = preprocess_image(path, sensor, j, width, height)

                        if j in training_ids:
                            if sensor == 'color':
                                training_color = np.append(training_color, image, axis=0)
                            if sensor == 'depthcolormap':
                                training_depth = np.append(training_depth, image, axis=0)
                            if sensor == 'icub_left':
                                training_icub_left = np.append(training_icub_left, image, axis=0)
                            if sensor == 'icub_right':
                                training_icub_right = np.append(training_icub_right, image, axis=0)
                            y_training.append((label_tool, label_action))

                        if j in validation_ids:
                            if sensor == 'color':
                                validation_color = np.append(validation_color, image, axis=0)
                            if sensor == 'depthcolormap':
                                validation_depth = np.append(validation_depth, image, axis=0)
                            if sensor == 'icub_left':
                                validation_icub_left = np.append(validation_icub_left, image, axis=0)
                            if sensor == 'icub_right':
                                validation_icub_right = np.append(validation_icub_right, image, axis=0)
                            y_validation.append((label_tool, label_action))

                        if j in testing_ids:
                            if sensor == 'color':
                                testing_color = np.append(testing_color, image, axis=0)
                            if sensor == 'depthcolormap':
                                testing_depth = np.append(testing_depth, image, axis=0)
                            if sensor == 'icub_left':
                                testing_icub_left = np.append(testing_icub_left, image, axis=0)
                            if sensor == 'icub_right':
                                testing_icub_right = np.append(testing_icub_right, image, axis=0)
                            y_testing.append((label_tool, label_action))

        print('concatenating of images' + ' ' + objectname + ' ' + 'done')
        # Remove the first row
        training_color, training_icub_right, training_icub_left = training_color[1:], training_icub_right[1:], training_icub_left[1:]
        training_depth = training_depth[1:]

        validation_color, validation_icub_right, validation_icub_left = validation_color[1:], validation_icub_right[1:], validation_icub_left[1:]
        validation_depth = validation_depth[1:]

        testing_color, testing_icub_right, testing_icub_left = testing_color[1:], testing_icub_right[1:], testing_icub_left[1:]
        testing_depth = testing_depth[1:]

        # Checking shapes
        print("training: ", training_color.shape, training_icub_right.shape, training_icub_left.shape, training_depth.shape)
        print("validation: ", validation_color.shape, validation_icub_right.shape, validation_icub_left.shape, validation_depth.shape)
        print("testing: ", testing_color.shape, testing_icub_right.shape, testing_icub_left.shape, testing_depth.shape)

        y_training = np.asarray(y_training, dtype=np.int32)
        y_validation = np.asarray(y_validation, dtype=np.int32)
        y_testing = np.asarray(y_testing, dtype=np.int32)

        # Checking shapes
        print("labels: ", y_training.shape, y_validation.shape, y_testing.shape)

        y_training = y_training[0::4]
        y_validation = y_validation[0::4]
        y_testing = y_testing[0::4]

        # Checking shapes again
        print("labels: ", y_training.shape, y_validation.shape, y_testing.shape)

        # Save data after processing each object
        save_data_to_disk(training_color, 'C:\Users\Frank\OneDrive\Bureaublad\ARC\deep-multimodal-learning\data\training_color_' + objectname + '.pkl')
        save_data_to_disk(training_icub_right, 'C:\Users\Frank\OneDrive\Bureaublad\ARC\deep-multimodal-learning\data\training_icub_right_' + objectname + '.pkl')
        save_data_to_disk(training_icub_left, 'C:\Users\Frank\OneDrive\Bureaublad\ARC\deep-multimodal-learning\data\training_icub_left_' + objectname + '.pkl')
        save_data_to_disk(training_depth, 'C:\Users\Frank\OneDrive\Bureaublad\ARC\deep-multimodal-learning\data\training_depth_' + objectname + '.pkl')

        save_data_to_disk(validation_color, 'C:\Users\Frank\OneDrive\Bureaublad\ARC\deep-multimodal-learning\data\validation_color_' + objectname + '.pkl')
        save_data_to_disk(validation_icub_right, 'C:\Users\Frank\OneDrive\Bureaublad\ARC\deep-multimodal-learning\data\validation_icub_right_' + objectname + '.pkl')
        save_data_to_disk(validation_icub_left, 'C:\Users\Frank\OneDrive\Bureaublad\ARC\deep-multimodal-learning\data\validation_icub_left_' + objectname + '.pkl')
        save_data_to_disk(validation_depth, 'C:\Users\Frank\OneDrive\Bureaublad\ARC\deep-multimodal-learning\data\validation_depth_' + objectname + '.pkl')

        save_data_to_disk(testing_color, 'C:\Users\Frank\OneDrive\Bureaublad\ARC\deep-multimodal-learning\data\testing_color_' + objectname + '.pkl')
        save_data_to_disk(testing_icub_right, 'C:\Users\Frank\OneDrive\Bureaublad\ARC\deep-multimodal-learning\data\testing_icub_right_' + objectname + '.pkl')
        save_data_to_disk(testing_icub_left, 'C:\Users\Frank\OneDrive\Bureaublad\ARC\deep-multimodal-learning\data\testing_icub_left_' + objectname + '.pkl')
        save_data_to_disk(testing_depth, 'C:\Users\Frank\OneDrive\Bureaublad\ARC\deep-multimodal-learning\data\testing_depth_' + objectname + '.pkl')

        save_data_to_disk(y_training, 'C:\Users\Frank\OneDrive\Bureaublad\ARC\deep-multimodal-learning\data\y_training_' + objectname + '.pkl')
        save_data_to_disk(y_validation, 'C:\Users\Frank\OneDrive\Bureaublad\ARC\deep-multimodal-learning\data\y_validation_' + objectname + '.pkl')
        save_data_to_disk(y_testing, 'C:\Users\Frank\OneDrive\Bureaublad\ARC\deep-multimodal-learning\data\y_testing_' + objectname + '.pkl')

if __name__ == '__main__':
    main()