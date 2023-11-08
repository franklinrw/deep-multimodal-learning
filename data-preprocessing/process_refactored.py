import numpy as np
import cv2
import pickle
import os
from functions import concatenate, normalize, resize, bgr_to_rgb

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

    data_sets = initialize_data_sets()

    for objectname in objectnames:
        for toolname in toolnames:
            for action in actions:
                label_action = actions.index(action)
                label_tool = toolnames.index(toolname)

                for sensor in sensornames:
                    path = os.path.join(base_path, objectname, toolname, action, sensor)

                    for j in range(10):
                        image = preprocess_image(path, sensor, j, width, height)

                        for set_name, data_set in data_sets.items():
                            if j in data_set['ids']:
                                data_set['data'][sensor].append(image)
                                data_set['labels'].append((label_tool, label_action))

                    for set_name, data_set in data_sets.items():
                        #data_set['data'][sensor] = np.vstack(data_set['data'][sensor])
                        #data_set['labels'] = np.array(data_set['labels'], dtype=np.int32)

                        filename = f"{set_name}.pkl"
                        object_folder = os.path.join('C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/data', objectname, toolname, action, sensor)

                        if not os.path.exists(object_folder):
                            os.makedirs(object_folder)
                        
                        save_path = os.path.join(object_folder, filename)
                        save_data_to_disk(data_set['data'][sensor], save_path)

                        # Save labels for the sensor
                        label_filename = f"y_{set_name}.pkl"
                        save_path = os.path.join(object_folder, label_filename)
                        save_data_to_disk(data_set['labels'], save_path)

            print(f"Processing of {objectname} done")
            data_sets = initialize_data_sets()

if __name__ == '__main__':
    main()
