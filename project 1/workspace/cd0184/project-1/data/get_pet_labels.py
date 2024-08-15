#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: 
# DATE CREATED:                                  
# REVISED DATE: 
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir

# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
# 
def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames 
    of the image files. These pet image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
         index 0 = pet image label (string)
    """
    #pet label list = in_files
    in_files = listdir(image_dir)
    #Iterate through the list to convert to lower case letters
    for filename in range(0, len(in_files),1):
        in_files[filename] = in_files[filename].lower()
    #Iterate through the list to split lower case string by '_'
    #create pet_name list
    pet_name_list = []
    for filename in in_files:
        split = filename.split("_")
        #print(split)
        pet_name = " "
        for name in split:
            if name.isalpha():
                pet_name += name + " "
        print(pet_name)
        pet_name_list.append(pet_name.strip())
     #Create empty dictionary named results_dic
    results_dic = dict()
    filename_list = listdir(image_dir)
    #Determine number of item in dictionary
    number_in_dict = len(results_dic)
    #Add new key-value pairs to dictionary ONLY when key doesn't already exist. This dictionary's value is a list #that contains only one item - the pet image label
   
    for key in range(0, len(filename_list), 1):
        if filename_list[key] not in results_dic:
            label_list = []
            label_list.append(pet_name_list[key])
            results_dic[filename_list[key]] = label_list
        else:
            print("Warning: Key = ", filename_list[key], "already exists in results_dic with value = ", results_dic[filename_list[key]])
            #Iterating through a dictionary printing all keys and their corresponding values
    #print("\Printing all key_value pairs in dictionary results_dic:")
    for key in results_dic:
        print("Filename = ", key, "Pet Label = ", results_dic[key])
    # Replace None with the results_dic dictionary that you created with this
    # function
    return results_dic