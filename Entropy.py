import numpy as np
import matplotlib.pyplot as plt
from math import log

# imports from keras 
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model, load_model 


# constant dimensions
HEIGHT = 30
WIDTH = 30 

# defining a blank image
image = np.zeros((HEIGHT,WIDTH), dtype='int8')

# defining the blocks 
block1 = [[0,0,1], [0,0,0], [1,0,1]]
block2 = [[0,1,0], [0,0,1], [1,0,0]]
block3 = [[0,0,1], [1,0,0], [0,1,1]]
block4 = [[1,0,1], [0,1,0], [1,0,0]]
block5 = [[1,0,0], [0,0,0], [0,0,1]]
#block6 = [[1,1,1], [0,1,0], [0,1,0]]
#block7 = [[0,1,0], [0,1,0], [1,0,1]]
block8 = [[0,0,0], [1,0,1], [0,1,0]]
#block9 = [[1,0,1], [1,1,1], [0,1,0]]

all_blocks = np.array([block1, block2, block3, 
                       block4, block5, block8], dtype='int8')

# defining all possible locations where the blocks can be placed in the image 
x_locations = list(range(1,HEIGHT-1,3))
y_locations = list(range(1,WIDTH-1,3))
all_possible_locations = [(x,y) for x in x_locations for y in y_locations]

# combining locations and blocks to give the complete range of possible actions
block_range = list(range(len(all_blocks)))
all_actions = [(x, y[0], y[1]) for x in block_range for y in all_possible_locations] 

# designing the conv. neural network

input_img = Input(shape=(HEIGHT, WIDTH,1))
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
x = Flatten()(x)
x = Dense(len(all_actions), activation = 'relu')(x)
output = Dense(len(all_actions))(x)

cnn = Model(input_img, output)
cnn.compile(optimizer='nadam', loss='mse') 

# Notes 
# Pooling layers not added as location information is important.
# The last layer does not have an activation to handle negative rewards.

# function to place a block in the x,y co-ordinates of the image. 
# returns a copy image. 
def place_block(image, block, x, y):
    x_start = x-1 
    x_stop = x+2
    y_start = y-1 
    y_stop = y+2
    
    res = np.copy(image)
    res[x_start:x_stop, y_start:y_stop] = block
    
    return res
        
# function to return log base 2
def ln(x):
    return log(x)/log(2)

# to get the entropy of a particular row/column in an image.
# Harcoding entropy as 1 for a row without any activated pixel as we want 
# the model to learn to draw stuff instead of leaving rows/column blank.
def get_slice_entropy(slce):
    n = len(slce)
    ones = np.count_nonzero(slce)
    if ones == 0:
        return 10
    else:
        zeros = n - ones
        p_ones = max(0.000001, ones / n)     # 0.000001 is added to avoid math error with log(0)/log(2)
        p_zeros = max(0.000001, zeros / n)   
        return (p_ones*ln(p_ones) + p_zeros*ln(p_zeros)) * -1

# to sum and return entropy for all rows and columns
def get_entropy(image):
    row_entropy = sum(np.apply_along_axis(get_slice_entropy, 0, image))
    col_entropy = sum(np.apply_along_axis(get_slice_entropy, 1, image))
    return row_entropy + col_entropy
    
# The reward is made inversely proportional to the entropy in the image,
# hence, directly proportional to the 'order' in the image.
# Also, count of non-zero elements ensures that the model achieves a higher reward 
# for filling in blank spaces in the image. 
def get_reward(image):
    score = np.count_nonzero(image) * 60 / get_entropy(image) 
    return score
    
# The reinforced training of the model can be seen as a game. 
# The game is initialized with a random block placed on a random location of a blank image. 
# The model then gets 'num_attempts' on the image to fill it with blocks iteratively. 
# The reward fed to the model should ensure the model learns to reduce entropy in the image,
# all while anticipating future rewards for the current action. 
def play_round(cnn, num_attempts, exploration=0.5):
    image = np.zeros((HEIGHT, WIDTH), dtype='int8')
    random_action = all_actions[np.random.randint(0,len(all_actions))]
    current_image = place_block(image, 
                                all_blocks[random_action[0]], 
                                random_action[1],
                                random_action[2])
    total_reward = 0
    decay_factor = 0.998
    locations_taken = []
    for _ in range(num_attempts):
        exploration *= decay_factor
        reshaped = current_image.reshape(1,HEIGHT, WIDTH,1)
        action_qs = cnn.predict(reshaped)
        
        # exploration vs exploitation for choosing the action
        if np.random.random() < exploration:
            chosen_action = np.random.randint(0,len(all_actions))
        else:
            chosen_action = np.argmax(action_qs)
        
        action_tuple = all_actions[chosen_action]
        location = (action_tuple[1], action_tuple[2])
        # penalized reward for overwriting a block
        if location in locations_taken:
            reward = -40
        else:
            locations_taken.append(location)
            # applying action to state/image
            new_image = place_block(current_image, 
                                    all_blocks[action_tuple[0]], 
                                    action_tuple[1],
                                    action_tuple[2])

            # calculating reward from applying the chosen action to the image
            # old_reward = get_reward(current_image)
            new_reward = get_reward(new_image) 

            # calculating future reward for the new image
            reshaped_new = new_image.reshape(1,HEIGHT, WIDTH,1)
            future_action = cnn.predict(reshaped_new)
            future_chosen_act = np.argmax(future_action)
            future_act_tuple = all_actions[future_chosen_act]
            future_location = (future_act_tuple[1], future_act_tuple[2])
            
            if future_location in locations_taken: # penalized reward for overwriting a block in the future
                future_reward = -40
            else:
                future_image = place_block(new_image,
                                      all_blocks[future_act_tuple[0]], 
                                      future_act_tuple[1],
                                      future_act_tuple[2])
                future_reward = get_reward(future_image) 
            
            # discounted future reward is added to the current reward for reinforcecment
            reward = new_reward + 0.8*future_reward
            
            
        # updating targets with the new reward
        target_action = action_qs
        target_action[0][chosen_action] = reward 
        
        # training with the new targets
        cnn.fit(reshaped, target_action, epochs=1, verbose=0)
        
        total_reward += reward  
        current_image = new_image
    
    avg_reward = total_reward / num_attempts
    return (avg_reward, current_image)

rewards = []
best_rewards = 0
lowest_entropy = 60
best_images = []
entropies = []
num_rounds = 400        # no of rounds
num_attempts = 100      # no of attempts in each round 

for _ in range(num_rounds): 
    avg_reward, current_image = play_round(cnn, num_attempts, 0.6)
    rewards.append(avg_reward)
    entropy = get_entropy(current_image)
    if (avg_reward > best_rewards) and (entropy < lowest_entropy):
        entropies.append(entropy)
        best_images.append(current_image)
        cnn.save('v1.h5')
        lowest_entropy = entropy
        best_rewards = avg_reward
            
