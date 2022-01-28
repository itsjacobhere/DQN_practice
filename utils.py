import numpy as np
import cv2
from matplotlib import pyplot as plt

"""

"""

class FrameStackingEnv:
    '''
    Wrapper for frame stacking an openAI env where the observation is in image format
    '''
    def __init__(self, env, width, height, num_stack = 4):
        
        self.env = env
        self.n = num_stack
        self.w = width
        self.h = height
        
        #self.action_space = env.action_space
        #self.observation_space = env.observation_space
        
        # array to store and stack images
        self.buffer = np.zeros((num_stack, height, width), dtype = 'uint8')
        self.frame = None
        
    def _preprocess_frame(self, frame):
        # resize image and convert to grayscale
        image = cv2.resize(frame, (self.w, self.h))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    
    def step(self, action):
        
        im, reward, done, info = self.env.step(action)
        self.frame = im.copy()
        im = self._preprocess_frame(im)
        
        # updating array with images of last n states
        self.buffer[1:self.n, :, :] = self.buffer[0:self.n-1, :, :]
        self.buffer[0, :, :] = im
        return self.buffer.copy(), reward, done, info
    
    
    def render(self, mode):
        if mode == 'rgb_array':
            return self.frame()
        return super(FrameStackingEnv, self).render(mode)
        
    
    @property
    def observation_space(self):
        return np.zeros((self.n, self.h, self.w))

    @property
    def action_space(self):
        return self.env.action_space
    
    def close(self):
        return self.env.close()
    
    def reset(self):
        
        # reset env and preprocess images
        im = self.env.reset()
        self.frame = im.copy()
        im = self._preprocess_frame(im)
        self.buffer = np.stack([im]*self.n, 0)
        return self.buffer.copy()
        
    def render(self, mode):
        self.env.render(mode)
        
        
if __name__ == '__main__':
    # test implementation
    
    import gym
    import time
    
    env = gym.make('Breakout-v0')
    env = FrameStackingEnv(env, 84, 84, num_stack=4)
    
    im = env.reset()
    
    
    print(env.observation_space.shape)
    print(env.action_space.n)
    
    ims = []
    for i in range(im.shape[-1]):
        
        print(i)
        ims.append(im[:,:,i])
        
    #plt.imshow(np.hstack(ims))
    #plt.show()
    time.sleep(1)
    
    
    env.step(1)
    
    for _ in range(10):
        
        im, r, d, info = env.step(np.random.randint(4))
        ims = []
        for i in range(im.shape[-1]):
            print(i)
            ims.append(im[:,:,i])
        #plt.figure(_)
        #plt.imshow(np.hstack(ims))
        #plt.show()        
        pass