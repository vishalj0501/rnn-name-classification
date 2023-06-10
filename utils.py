import io
import os 
import torch
import random
from unidecode import unidecode
import glob


all_letters="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'"
n_letters=len(all_letters)

def unicode_to_ascii(name):
    return unidecode(name)

# print(unicode_to_ascii('kožušček'))

def load_data():
    category_lines={}
    all_categories=[]
    # def find_files(path):
    #     return glob.glob(path)
    def read_lines(filename):
        lines=io.open(filename,encoding='utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]
    for filename in glob.glob('data/names/*.txt'):
        category=os.path.splitext(os.path.basename(filename))[0] 
        all_categories.append(category)
        lines=read_lines(filename)
        category_lines[category]=lines
    return category_lines,all_categories


def letter_to_index(letter):
    all_letters="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'"
    return all_letters.find(letter)

def letter_to_tensor(letter):
    tensor=torch.zeros(1,57)
    tensor[0][letter_to_index(letter)]=1
    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, 57)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor

def random_training_example(category_lines, all_categories):
    
    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]
    
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


# random_training_example()


# print(line_to_tensor('Jones'))

# with open ('test.txt', 'w') as file:  
#     file.write(str(line_to_tensor('Jones')))  

# print(letter_to_tensor('a'))

# letter_to_index('a')