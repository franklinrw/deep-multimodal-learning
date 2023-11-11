import pickle
import os
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

                # New structure: list of tuples (image, label)
                data_with_labels = []

                for sensor in sensornames:
                    path = os.path.join(base_path, objectname, toolname, action, sensor)

                    for j in range(10):
                        image = preprocess_image(path, sensor, j, width, height)

                        # Append the image and its label as a tuple
                        label = (label_tool, label_action)
                        data_with_labels.append((image, label))

                    for set_name in ['training', 'validation', 'testing']:
                        object_folder = os.path.join('C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/data_v2', objectname, toolname, action, sensor)

                        if not os.path.exists(object_folder):
                            os.makedirs(object_folder)

                        filename = f"{set_name}_{sensor}.pkl"
                        save_path = os.path.join(object_folder, filename)

                        # Save the data_with_labels list to a pickle file
                        with open(save_path, 'wb') as f:
                            pickle.dump(data_with_labels, f)

            print(f"Processing of {objectname} done")

if __name__ == '__main__':
    main()
