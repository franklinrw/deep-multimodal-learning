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

                # Collecting data for all sensors together
                all_data_combined = []

                for j in range(10):  # Assuming each sensor has 10 images
                    sensor_data = []
                    for sensor in sensornames:
                        path = os.path.join(base_path, objectname, toolname, action, sensor)
                        image = preprocess_image(path, sensor, j, width, height)
                        label = (label_tool, label_action)
                        sensor_data.append((image, label))
                    all_data_combined.append(sensor_data)

                # Shuffle and split the combined data
                random.shuffle(all_data_combined)
                split1 = int(len(all_data_combined) * 0.6)
                split2 = split1 + int(len(all_data_combined) * 0.2)
                train_set, val_set, test_set = all_data_combined[:split1], all_data_combined[split1:split2], all_data_combined[split2:]

                # Separate the data for each sensor and save
                for sensor_index, sensor in enumerate(sensornames):
                    train_set_sensor = [item[sensor_index] for item in train_set]
                    val_set_sensor = [item[sensor_index] for item in val_set]
                    test_set_sensor = [item[sensor_index] for item in test_set]

                    def save_data(dataset, set_name):
                        object_folder = os.path.join('C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/fusion', objectname, toolname, action, sensor)
                        if not os.path.exists(object_folder):
                            os.makedirs(object_folder)
                        filename = f"{set_name}.pkl"
                        save_path = os.path.join(object_folder, filename)
                        with open(save_path, 'wb') as f:
                            pickle.dump(dataset, f)

                    # Save datasets for each sensor
                    save_data(train_set_sensor, 'training')
                    save_data(val_set_sensor, 'validation')
                    save_data(test_set_sensor, 'testing')

            print(f"Processing of {objectname} done")

if __name__ == '__main__':
    main()
