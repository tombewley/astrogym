import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


plt.rcParams["toolbar"] = "None"


class AstroGymEnv(gym.Env):
    """
    OpenAI Gym-compatible environment for exploring astronomical images.
    """

    render_size = 100 # In pixels
    min_window_size = 100 # In pixels; CV2 will interpolate if this is < render_size
    plt_window_size = (6, 6) # In inches
    percentile_clip = 99 # Brightness percentile to clip each channel at

    def __init__(self, img, do_render=False):
        self.observation_space = None
        self.action_space = gym.spaces.Box(np.float32(-1), np.float32(1), shape=(3,)) 
        ext = img.split(".")[-1]
        if ext == "jpg": self.img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        elif ext == "npy": self.img = np.load(img)
        self.img_size = self.img.shape[0]
        assert self.img_size == self.img.shape[1], "Image must be square"
        assert self.img_size >= 4 * self.render_size, "Insufficient image size" # NOTE: This is completely arbitrary
        self.img = np.clip(self.img, a_min=None, a_max=np.percentile(self.img, self.percentile_clip))
        self.img /= self.img.max()
        self._state = (0, self.img_size, 0, self.img_size)
        self.brightness_baseline = self.obs().sum()
        self.do_render = do_render
        if self.do_render: 
            self.fig, self.ax = plt.subplots(figsize=self.plt_window_size)
            self.fig.canvas.manager.set_window_title("AstroGym")
            self.ax.set_xticks([]); self.ax.set_yticks([]); plt.ion(); self.ax.set_aspect("equal", "box")
            self.ax.margins(x=0, y=0)
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)            
            self._img_plt = self.ax.imshow([[0]], extent=(-1,1,1,-1))
            self._action_indicator = (
                Circle(xy=(0, 0), radius=0.05, facecolor="w", alpha=0.5),
                Rectangle(xy=(-1, -1), width=2, height=2, edgecolor="w", fill=False)
                )
            for element in self._action_indicator: self.ax.add_artist(element)

    @property 
    def num_channels(self): return self.img.shape[2]
    
    def reset(self):
        self._state = (0, self.img_size, 0, self.img_size)
        self._action = None
        self._obs = self.obs()
        return self._obs 
    
    def step(self, action):
        assert action in self.action_space
        self._action = action
        # TODO: This implementation could be tidier; use NumPy?
        # Only 3DoF here; self._state should be (x, y, w).
        xl_old, xu_old, yl_old, yu_old = self._state
        x = (xl_old + xu_old) / 2
        y = (yl_old + yu_old) / 2
        w = xu_old - xl_old
        assert w == yu_old - yl_old
        x += self._action[0] * w / 2
        y += self._action[1] * w / 2
        w = round(max(min(w + self._action[2] * w / 2, self.img_size), self.min_window_size))
        xl, xu = x - w / 2, x + w / 2
        if xl < 0: 
            xl, xu = 0, w
        elif xu > self.img_size: 
            xl, xu = self.img_size - w, self.img_size
        yl, yu = y - w / 2, y + w / 2
        if yl < 0: 
            yl, yu = 0, w
        elif yu > self.img_size: 
            yl, yu = self.img_size - w, self.img_size
        xl, xu, yl, yu = int(round(xl)), int(round(xu)), int(round(yl)), int(round(yu))
        assert xu - xl == yu - yl == int(w)
        self._state = (xl, xu, yl, yu)
        if xl != xl_old or xu != xu_old or yl != yl_old or yu != yu_old:
            self._obs = self.obs() # To save computation, only redo observation if it has changed.
        return self._obs, self.reward(), self.done(), {}

    def obs(self): 
        xl, xu, yl, yu = self._state
        return cv2.resize(self.img[yl:yu, xl:xu], (self.render_size, self.render_size))

    def reward(self):  
        """
        Experimental reward function for debugging: change in mean brightness vs full image.
        """   
        return (self._obs.sum() - self.brightness_baseline) / (self.render_size**2)

    def done(self):         
        return False

    def render(self, mode="human", pause=1e-6):
        if mode == "human": 
            assert self.do_render, "Not set up for rendering; initialise with do_render=True"
            # NOTE: Set up for five-channel images with BGR as the middle three channels.
            self._img_plt.set_data(cv2.cvtColor(self._obs[:,:,1:4], cv2.COLOR_BGR2RGB))
            if self._action is not None:
                self._action_indicator[0].center = self._action[:2]
                w = 2 + self._action[2]
                self._action_indicator[1].xy = self._action[:2] - w / 2
                self._action_indicator[1].set_width(w)
                self._action_indicator[1].set_height(w)
            plt.pause(pause)
        elif mode == "rgb_array": 
            return self._obs 