import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def rotate(x_val, y_val, rotation):
    current_theta = np.arctan((y_val)/(x_val))
    new_theta = current_theta + rotation
    new_x = x_val * np.cos(new_theta) - y_val * np.sin(new_theta)
    new_y = x_val * np.sin(new_theta) + y_val * np.cos(new_theta) # Check radians vs degrees

    return new_x, new_y

def augment_image(image, rotation):
    # print("Performing image augmentation...") # DEBUG
    
    # Rotation
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, 1) # -8 <= rotation <= 8
    rotated = cv2.warpAffine(image, M, (cols, rows))

    # Flipping
    flipped = cv2.flip(image, 1)  # 1 for horizontal flip
    
    # Brightness adjustment
    bright = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    
    # Add noise
    noise = np.random.normal(0, 0.3, image.shape).astype(np.uint8)
    noisy = cv2.addWeighted(image, 1, noise, 0.5, 0)
    
    # print("Augmentation complete.") # DEBUG
    return rotated, flipped, bright, noisy

train_images = os.listdir("datasets/robot_detection/train/images/")
# Example usage
for image_name in train_images:
    image = cv2.imread("datasets/robot_detection/train/images/" + image_name)

    if image is None:
        print(f"Failed to load image {image_name}")
    else:
        rotation = np.random.choice([1, -1]) * np.random.rand() * 8
        rotated, flipped, bright, noisy = augment_image(image, rotation)

        # Save results

        # Images
        cv2.imwrite("datasets/robot_detection/train/images_altered/" + image_name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cv2.imwrite("datasets/robot_detection/train/images_altered/rotated_" + image_name, cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        cv2.imwrite("datasets/robot_detection/train/images_altered/bright_adjust_" + image_name, cv2.cvtColor(bright, cv2.COLOR_BGR2RGB))
        cv2.imwrite("datasets/robot_detection/train/images_altered/noise_" + image_name, cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))

        # Labels
        # "<class> <x_center> <y_center> <width> <height>" Coordinates must be normalized where max is 1, lowest is 0.
        with open("datasets/robot_detection/train/labels/" + image_name.replace(".jpg", ".txt"), "r") as file_in:
            with open("datasets/robot_detection/train/labels_altered/rotated_" + image_name.replace(".jpg", ".txt"), "w") as file_out:
                x_center = 0
                y_center = 0
                width = 0
                height = 0
                decimal = False
                for line in file_in:
                    state = 0
                    current = 0
                    current_decimal = []
                    for i in line:
                        if i != " " and i != "\n":
                            if i == ".":
                                decimal = True

                            elif decimal == True:
                                current_decimal.append(i)
                        else:
                            a = 1
                            for x in current_decimal: # Convert string read to decimal we can use
                                current += (float)(x) * (float)(10**(-a))
                                a += 1

                            if state == 0:
                                file_out.write("0 ")

                            elif state == 1:
                                x_center = current
                            elif state == 2:
                                y_center = current
                                new_x, new_y = rotate(x_center - 0.5, y_center - 0.5, rotation)
                                new_x += 0.5
                                new_y += 0.5

                                file_out.write(f"{new_x} {new_y} ")
                            elif state == 3:
                                width = current
                            elif state == 4:
                                height = current

                                # Find old top left and bottom right corner coordinates
                                new_top_left_x, new_top_left_y = rotate(x_center - width/2 - 0.5, y_center + height/2 - 0.5, rotation)
                                new_top_left_x += 0.5
                                new_top_left_y += 0.5

                                new_bottom_right_x, new_bottom_right_y = rotate(x_center + width/2 - 0.5, y_center - height/2 - 0.5, rotation)
                                new_bottom_right_x += 0.5
                                new_bottom_right_y += 0.5

                                new_width = np.abs(new_bottom_right_x - new_top_left_x)
                                new_height = np.abs(new_top_left_y - new_bottom_right_y)

                                file_out.write(f"{new_width} {new_height}\n")
                                state = 0
                                decimal = False
                                current = 0
                                break
                                
                            state += 1
                            decimal = False
                            current = 0

        with open("datasets/robot_detection/train/labels/" + image_name.replace(".jpg", ".txt"), "r") as file_in:
            with open("datasets/robot_detection/train/labels_altered/" + image_name.replace(".jpg", ".txt"), "w") as file_out:
                for line in file_in:
                    file_out.write(line)
        with open("datasets/robot_detection/train/labels/" + image_name.replace(".jpg", ".txt"), "r") as file_in:
            with open("datasets/robot_detection/train/labels_altered/bright_adjust_" + image_name.replace(".jpg", ".txt"), "w") as file_out:
                for line in file_in:
                    file_out.write(line)
        with open("datasets/robot_detection/train/labels/" + image_name.replace(".jpg", ".txt"), "r") as file_in:
            with open("datasets/robot_detection/train/labels_altered/noise_" + image_name.replace(".jpg", ".txt"), "w") as file_out:
                for line in file_in:
                    file_out.write(line)

        # # Display the results
        # plt.figure(figsize=(20, 5))
        
        # plt.subplot(151)
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.title('Original')
        # plt.axis('off')
        
        # plt.subplot(152)
        # plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        # plt.title('Rotated')
        # plt.axis('off')
        
        # plt.subplot(153)
        # plt.imshow(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))
        # plt.title('Flipped')
        # plt.axis('off')
        
        # plt.subplot(154)
        # plt.imshow(cv2.cvtColor(bright, cv2.COLOR_BGR2RGB))
        # plt.title('Brightness Adjusted')
        # plt.axis('off')
        
        # plt.subplot(155)
        # plt.imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
        # plt.title('Noisy')
        # plt.axis('off')
        
        # plt.tight_layout()
        # plt.show()

print("Script completed")