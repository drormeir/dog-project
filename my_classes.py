# I decided to create a data generator for the dogs images from several reasons:

# 1. I don't have enough memory to hold all the dogs images for the training process.

# 2. I want to be able to train the network on smaller data (fewer labels to classify) in order to speed up the development phase of this project.

# 3. In my opinion, there are too few samples for each class to train from (~6680 images divided to 133 classes), I want to use data augmentation

# 4. The training data is not balanced, some classes have ~70 images while others have ~25, I wish to balance the images number for each class, using the data augmentation.

# 5. Using any sub type of gradient descent with a mini batch, has an advantage of faster computation due to paralelization, but may converge faster (efficiently) than a full batch gradient descent, only when the whole spectrum of the training data has a good represention in the small random sampling of the mini batch. However, in this project, the number of classes is pretty high (133) which makes a simple naive uniform random sampling to be highly unbalanced with respect to the classes labels in each mini batch step, even when using a mini batch size of 256. This leads to inefficiency in the converging process.
#    I address this mini sampling issue inside the data generator by using a Randomized Round Robin algorithm for sampling the mini batch: First I randomly choose the next label, and from that label I randomly choose the next image. Please look in the code for further details.

# For implementing the data generator I followed the recommandations found at:
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# I copied some parts from the original dogs projects notebook that deals with image pre-loading

import cv2
import numpy as np
from keras.utils import np_utils
from keras.preprocessing import image                  
from PIL import ImageFile                            
import os
import math
from tqdm import tqdm

class BalancedImageGenerator(object):
    def __init__(self, images_paths, images_labels, save_to_dir, image_size=(100, 100), batch_size = 32, \
                 augmentation_level=2.0, specific_labels=[], \
                 width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, rotation_range=30, horizontal_flip=True):
        print('BalancedImageGenerator started...')
        os.makedirs(save_to_dir, exist_ok=True)
        self.image_size  = image_size
        self.image_shape = image_size + (3,)
        self.__init_original_images_paths(images_paths, images_labels, specific_labels)
        num_original_images = self.count_total_images()
        max_num_files_per_label, most_common_label, min_num_files_per_label, least_common_label\
        = self.find_extreme_num_common_label()
        if augmentation_level >= 1e-6:
            target_augmented_num = round(self.calc_num_most_common_label() * augmentation_level)
        else:
            target_augmented_num = 0
        self.reproducible_random()
        datagen = image.ImageDataGenerator(rotation_range=rotation_range, width_shift_range=width_shift_range, \
                                           height_shift_range=height_shift_range, zoom_range=zoom_range, \
                                           horizontal_flip=horizontal_flip)
        for label in tqdm(self.images_paths):
            self.__augment_images(label, target_augmented_num, datagen, save_to_dir)
        self.reset_round_robin()
        self.batch_size = batch_size
        print('BalancedImageGenerator is initialized successfully!')
        print('Number of unique labels (classes):',self.num_classes())
        print('Single image shape:',self.image_shape)
        print('Total number of original images used:', num_original_images)
        print('Most common label is:',most_common_label,'with',max_num_files_per_label,'occurrences')
        print('Least common label is:',least_common_label,'with',min_num_files_per_label,'occurrences')
        imbalance_ratio = ((max_num_files_per_label - min_num_files_per_label) * 100.) / max_num_files_per_label
        print('Original imbalance ratio: %.2f%%' % imbalance_ratio)
        print('Total number of images after augmentation:',self.count_total_images())
        print('Final average number of images per label:',self.count_total_images()/self.num_classes())
        print('Batch size:',self.batch_size)
        print('Possible number of batch steps per epoch: minimum:',self.min_steps_per_epoch(),\
              'maximum:',self.max_steps_per_epoch())
        print('Suggested number of batch steps per epoch:',self.steps_per_epoch())
        

    def batch_maker(self):
        # Infinite loop
        while True:
            list_of_tensors = []
            labels = []
            for i in range(self.batch_size):
                if self.ind_round_robin_labels >= self.round_robin_labels.size:
                    np.random.shuffle(self.round_robin_labels)
                    self.ind_round_robin_labels = 0
                label = self.round_robin_labels[self.ind_round_robin_labels]
                self.ind_round_robin_labels += 1
                ind_round_robin_file = self.ind_round_robin_files[label]
                if ind_round_robin_file >= self.images_paths[label].size:
                    np.random.shuffle(self.images_paths[label])
                    ind_round_robin_file = 0
                self.ind_round_robin_files[label] = ind_round_robin_file + 1
                image_path = self.images_paths[label][ind_round_robin_file]
                list_of_tensors.append(self.__path_to_tensor4D(image_path,False))
                labels.append(label)
            yield self.__stack(list_of_tensors, labels)

    def max_steps_per_epoch(self):
        num_images_per_step = self.count_total_images()
        result = num_images_per_step // self.batch_size
        if num_images_per_step % self.batch_size:
            result += 1
        return result
    
    def min_steps_per_epoch(self):
        num_images_per_step = min(self.num_classes(),self.count_total_images())
        result = num_images_per_step // self.batch_size
        if num_images_per_step % self.batch_size:
            result += 1
        return result
    
    def med_steps_per_epoch(self):
        return (self.max_steps_per_epoch() + self.min_steps_per_epoch()) // 2
    
    # use at least "labels_multiplier" images from each label per epoch step
    def steps_per_epoch(self, labels_multiplier=10):
        num_labels = self.round_robin_labels.size
        num_images = self.count_total_images()
        num_images_per_step = min(num_images,num_labels*max(labels_multiplier,1))
        result = num_images_per_step // self.batch_size
        if num_images_per_step % self.batch_size:
            result += 1
        return result

    def reset_round_robin(self):
        self.reproducible_random()
        round_robin_labels = []
        self.ind_round_robin_files = {}
        for label in self.images_paths:
            self.images_paths[label] = np.sort(self.images_paths[label])
            self.ind_round_robin_files[label] = self.images_paths[label].size
            round_robin_labels.append(label)
        self.round_robin_labels = np.array(round_robin_labels)
        self.ind_round_robin_labels = self.round_robin_labels.size
        
    def calc_num_most_common_label(self):
        max_num_files_per_label = 0
        for label in self.images_paths:
            num_file_names = self.images_paths[label].size
            if num_file_names > max_num_files_per_label:
                max_num_files_per_label = num_file_names
        return max_num_files_per_label

    def find_extreme_num_common_label(self):
        max_num_files_per_label = int(0)
        min_num_files_per_label = int(1e9)
        most_common_label = -1
        least_common_label = -1
        for label in self.images_paths:
            num_file_names = self.images_paths[label].size
            if num_file_names > max_num_files_per_label:
                max_num_files_per_label = num_file_names
                most_common_label = label
            elif num_file_names < min_num_files_per_label:
                min_num_files_per_label = num_file_names
                least_common_label = label
        return max_num_files_per_label, most_common_label, min_num_files_per_label, least_common_label
    
    def count_total_images(self):
        count = 0
        for label in self.images_paths:
            num_file_names = self.images_paths[label].size
            count += num_file_names
        return count

    def reproducible_random(self):
        # using reproducible result
        # https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
        np.random.seed(42)

    def filter_tensors(self, tensors, all_labels):
        labels, indices = self.__find_relevant_labels(all_labels)
        return tensors[indices], self.__to_categorical(labels)
        
    def paths_to_tensor4D(self, images_paths, images_labels):
        if images_labels is None:
            labels = None
            paths = images_paths
        else:
            labels, indices = self.__find_relevant_labels(images_labels)
            paths           = images_paths[indices]
        if type(paths) == str:
            # convert a single instance into a list because later I use "for" loop
            paths = [paths]
        if len(paths) > 2:
            list_of_tensors = [self.__path_to_tensor4D(img_path,True) for img_path in tqdm(paths)]
        else:
            list_of_tensors = [self.__path_to_tensor4D(img_path,True) for img_path in paths]
        return self.__stack(list_of_tensors, labels)
    
    def num_classes(self):
        return self.round_robin_labels.size

    # Not all images of the same label will be resampled in the same amount
    def __augment_images(self, label, target_augmented_num, datagen, save_to_dir):
        original_num = self.images_paths[label].size
        if target_augmented_num < 1:
            # do not augment original images
            target_augmented_num = original_num
            residue = 0
        elif target_augmented_num < original_num:
            # select a subset from original images
            np.random.shuffle(self.images_paths[label])
            self.images_paths[label] = self.images_paths[label][:target_augmented_num]
            original_num = target_augmented_num
            residue = 0
        else:
            residue = target_augmented_num % original_num
            if residue > 0:
                np.random.shuffle(self.images_paths[label])
        images_paths   = self.images_paths[label]
        prefix_path    = save_to_dir + 'augmented_' + str(label) + '_'
        new_paths      = []
        for i in range(original_num):
            # read original image
            image_path = images_paths[i]
            x = self.__path_to_tensor4D(image_path,True)
            image_extension = os.path.splitext(image_path)[1]
            new_path = prefix_path + str(i) + '_original_cv2' + image_extension
            self.__tensor4D_to_path(x,new_path)
            new_paths.append(new_path)
            # consider the example of original_num == 10 and target_augmented_num == 17
            # I want to calculate the number of addtional images for each original instance
            # Hence, for the first 7 original images the calculation should return 1 and for the last 3 it will return 0
            num_additional = target_augmented_num // original_num
            if i >= residue:
                num_additional -= 1
            if num_additional < 1:
                # continue to pre process next image even if there is no additional augmentation
                continue
            for new_x in datagen.flow(x, batch_size=1):
                # maintain original file name extension for saving in the original image file format
                new_path = prefix_path + str(i) + '_' + str(num_additional) + image_extension
                # do not use ImageDataGenerator automatic save because it uses hash number which can be duplicated
                self.__tensor4D_to_path(new_x,new_path)
                new_paths.append(new_path)
                num_additional -= 1
                if num_additional < 1:
                    break
        self.images_paths[label] = np.array(new_paths)

    def __init_original_images_paths(self, images_paths, images_labels, specific_labels):
        all_labels = self.__get_all_labels(images_labels)
        unique_labels, indexes_in_unique = np.unique(all_labels, return_inverse = True)
        # now I have: unique_labels[indexes_in_unique] == all_labels
        # init images paths dictionary with label as a primary key
        self.images_paths = {}
        self.labels_to_index = {}
        ind = 0
        for i,label in enumerate(unique_labels):
            if (len(specific_labels) > 0) and (label not in specific_labels):
                continue
            self.images_paths[label] = images_paths[np.where(indexes_in_unique == i)]
            self.labels_to_index[label] = ind
            ind += 1

            
            
    def __path_to_tensor4D(self, img_path, histEqual):
        if histEqual:
            # source code for histogram equalization taken from:
            # https://www.packtpub.com/mapt/book/application_development/9781785283932/2/ch02lvl1sec26/enhancing-the-contrast-in-an-image
            img = cv2.imread(img_path)

            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

            # equalize the histogram of the Y channel
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

            # convert the YUV image back to RGB format
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
            img = cv2.resize(img,self.image_size)
        else:
            ImageFile.LOAD_TRUNCATED_IMAGES = True                 
            # loads RGB image as PIL.Image.Image type
            img = image.load_img(img_path, target_size=self.image_size)
            # convert PIL.Image.Image type to 3D tensor with pre defined shape
            img = image.img_to_array(img)
        # convert 3D tensor to 4D tensor
        return np.expand_dims(img, axis=0)

    def __tensor4D_to_path(self, x, img_path):
        img = image.array_to_img(np.squeeze(x, axis=0))
        img.save(img_path)

    def __find_relevant_labels(self, labels):
        all_labels      = self.__get_all_labels(labels)
        indices         = np.where(np.isin(all_labels, self.round_robin_labels))[0]
        return all_labels[indices], indices
    
    def __stack(self, list_of_tensors, labels):
        # here can come any future pre process
        # I failed to create my own keras layer that knows how to cast values from int32 to float32
        # So I decided to do it here
        res_tensors = np.vstack(list_of_tensors).astype('float32')/255.0
        return res_tensors, self.__to_categorical(labels)
    
    def __to_categorical(self, labels):
        if labels is None:
            return None
        # This container may hold a subset of the original labels
        ind_labels = [self.labels_to_index[label] for label in labels]
        return np_utils.to_categorical(np.array(ind_labels), self.num_classes())
    
    def __get_all_labels(self, images_labels):
        if type(images_labels) is not np.ndarray:
            all_labels = np.array(images_labels)
        elif images_labels.size == images_labels.shape[0]:
            # labels are vector
            all_labels  = images_labels
        else:
            # images labels are one-hot matrix
            all_labels  = np.argmax(images_labels,axis=1)
        return all_labels
    
        