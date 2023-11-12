import pickle
import os
import random
from functions import preprocess_image

### FOLDER LABELS ###
sensornames = ['color', 'depthcolormap', 'icub_left', 'icub_right']
toolnames = ['hook', 'ruler', 'spatula', 'sshot']
actions = ['left_to_right', 'pull', 'push', 'right_to_left']
objectnames = ['0_woodenCube', '1_pearToy', '2_yogurtYellowbottle', '3_cowToy', '4_tennisBallYellowGreen',
            '5_blackCoinbag', '6_lemonSodaCan', '7_peperoneGreenToy', '8_boxEgg','9_pumpkinToy',
            '10_tomatoCan', '11_boxMilk', '12_containerNuts', '13_cornCob', '14_yellowFruitToy',
            '15_bottleNailPolisher', '16_boxRealSense', '17_clampOrange', '18_greenRectangleToy', '19_ketchupToy']

def main():
    width, height = 256, 192
    base_path = 'C:/Users/Frank/OneDrive/Bureaublad/action_recognition_dataset/'

    for objectname in objectnames:
        for toolname in toolnames:
            for action in actions:
                label_action = actions.index(action)
                label_tool = toolnames.index(toolname)

                for sensor in sensornames:
                    path = os.path.join(base_path, objectname, toolname, action, sensor)

                    # Collect all images for this sensor
                    all_images = []
                    for j in range(10):
                        image = preprocess_image(path, sensor, j, width, height)
                        label = (label_tool, label_action)
                        all_images.append((image, label))

                    # Shuffle and split data into 60/20/20
                    random.shuffle(all_images)
                    split1 = int(len(all_images) * 0.6)
                    split2 = split1 + int(len(all_images) * 0.2)
                    train_set, val_set, test_set = all_images[:split1], all_images[split1:split2], all_images[split2:]

                    # Define a helper function to save data
                    def save_data(dataset, set_name):
                        object_folder = os.path.join('C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/data_v2', objectname, toolname, action, sensor)
                        if not os.path.exists(object_folder):
                            os.makedirs(object_folder)
                        filename = f"{set_name}.pkl"
                        save_path = os.path.join(object_folder, filename)
                        with open(save_path, 'wb') as f:
                            pickle.dump(dataset, f)

                    # Save the datasets
                    save_data(train_set, 'training')
                    save_data(val_set, 'validation')
                    save_data(test_set, 'testing')

            print(f"Processing of {objectname} done")

if __name__ == '__main__':
    main()


# ... [existing imports and folder labels] ...

def main():
    width, height = 256, 192
    base_path = 'C:/Users/Frank/OneDrive/Bureaublad/action_recognition_dataset/'

    # Define depth range (replace these with your actual min and max depth values)
    min_depth = 14
    max_depth = 113

    for objectname in objectnames:
        for toolname in toolnames:
            for action in actions:
                label_action = actions.index(action)
                label_tool = toolnames.index(toolname)

                for sensor in sensornames:
                    path = os.path.join(base_path, objectname, toolname, action, sensor)

                    all_images = []
                    for j in range(10):
                        # Check if the sensor is a depth sensor
                        is_depth_sensor = 'depth' in sensor

                        # Pass min and max depth if it's a depth sensor
                        image = preprocess_image(path, sensor, j, width, height, 
                                                min_depth if is_depth_sensor else None, 
                                                max_depth if is_depth_sensor else None)
                        label = (label_tool, label_action)
                        all_images.append((image, label))

                    # [Shuffle and split data, save data as in your original script]

            print(f"Processing of {objectname} done")

if __name__ == '__main__':
    main()
