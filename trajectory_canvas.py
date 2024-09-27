# In[0]:
import torch
from ipycanvas import Canvas
from ipywidgets import Button
from load_generative_model import torch
from typing import Callable

class trajectory_canvas:
    """Canvas object for displaying and manipulating line-graph 'trajectory' representations of slices from a tensor.

    Typical use:
        canvas = trajectory_canvas(multi_dimensional_trajectory_data, mouseoff_function, dimension_to_display: int)
        app = VBox([canvas]) 
        display(app) 

    Parameters: 
    trajectory_data: n x 1-dimensional time-series data
    refresh_function: a function which is called everytime the graph data is updated (eg mouseoffs)
    latent_dimension: the dimension < n of the data to be represented in this graph
       
    
    Attributes:
        

    Methods:
       

    """
    def __init__(self, trajectory_data: torch.Tensor, refresh_function: Callable, latent_dimension: int):

        # Slice tensor
        self.trajectory_data = trajectory_data[0,latent_dimension,:]
        self.latent_dimension = latent_dimension

        # Get info about input tensor
        self.data_min: float = min(torch.min(self.trajectory_data, dim=0, keepdim=True)[0].item(),3)
        self.data_max: float = max(torch.max(self.trajectory_data, dim=0, keepdim=True)[0].item(),-3)
        self.data_len: int = self.trajectory_data.size(dim=0)

        # Canvas size in pixels
        self.canvas_size: int = 400
        self.canvas_width: int = self.canvas_size
        self.canvas_height: int = self.canvas_size

        # the amount of nearby data that is changed. Proportional to the amount of data/resolution of canvas
        self.resolution: int = int(self.canvas_size/self.data_len)
        
        # Create a canvas for drawing
        self.canvas = Canvas(width=self.canvas_width, height=self.canvas_height)
        # self.canvas.scale(1, -1) #flip canvas so y is the right way around
        self.canvas.stroke_style = 'black'
        self.canvas.line_width = 1
        
        # # Initialize button to clear the canvas
        # self.clear_button = Button(description='Clear')
        # self.clear_button.on_click(self.clear_canvas)

        # Flag to track whether the mouse is drawing
        self.is_drawing = False

        # Assign mouse event handlers to the canvas
        self.canvas.on_mouse_move(self.on_mouse_move)
        self.canvas.on_mouse_down(self.on_mouse_down)
        self.canvas.on_mouse_up(self.on_mouse_up)
        self.refresh_function = refresh_function

        self.refresh()

    # Re-implementing the map function from arduino etc to put tensors in nice range and back
    def __tensor_map(self, input: torch.Tensor, in_min:float, in_max:float, out_min:float, out_max:float):
        """Map tensor values to a range
        """
        if (in_max == in_min):
            return torch.full(input.size(),-1)
        else:
            return (input - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
    def __tensor_interpolate(self, input: torch.Tensor, output_size: int):
        """Resize tensor while keeping the shape of the data (1D up/downscaling)
        """
        # Interpolate tensor to match gui
        output: torch.Tensor = torch.nn.functional.interpolate(input.unsqueeze(0).unsqueeze(0),output_size, mode = 'linear' ).squeeze().squeeze()
        return output

    # Function to draw the tensor on the canvas
    def draw_tensor(self):
        """"Update data and plot it as a path 
        """
        # Sync frontend with backend data
        self.canvas_trajectory: torch.Tensor = self.__tensor_interpolate(self.__tensor_map(self.trajectory_data, self.data_min, self.data_max, 0, self.canvas_height),self.canvas_width)

        self.canvas.clear()  # Clear previous drawings
        self.canvas.begin_path() # Start Drawing
        
        # tensor_size: int = self.trajectory_data.size(dim=0)
        
        # Draw simple path of points
        self.canvas.begin_path()
        self.canvas.move_to(0, int(self.canvas_trajectory[0]))


        # canvas.fill()
        
        for x in range(self.canvas_width):
            y = int(self.canvas_trajectory[x]) 
            if y>0:
                self.canvas.line_to(x, y)
            # canvas.fill_rect(x * (canvas_width // tensor_size), y, pixel_size*2, pixel_size)

        self.canvas.stroke()

    # Frontend handlers - Eg: event listeners for mouse movements to draw. Only operates on canvas data
    def on_mouse_down(self, x, y):
        """Edit data in canvas by drawing with mouse. Works for initial click.
        """
        self.is_drawing = True
        if 0 <= x < self.canvas_width and 0 <= y < self.canvas_height:
            # Update the tensor with the y-coordinate (height) at position x
            self.canvas_trajectory[int(x)] = y
            self.canvas.begin_path()
            self.canvas.move_to(x, y)

    def on_mouse_move(self, x: int, y: int):
        """Edit data in canvas by drawing with mouse. Works for mouse move
        """
        if self.is_drawing:
            if 0 <= x < self.canvas_width and 0 <= y < self.canvas_height:
                if y != self.canvas_trajectory[int(x)]:
                    # Erase
                    self.canvas.fill_style = 'white'
                    self.canvas.fill_rect(x, int(self.canvas_trajectory[int(x)]), self.resolution*2, self.canvas_height )
        
                    # Update the tensor with the y-coordinate (height) at position x and its neighbours
                    for index in range(int(x)-self.resolution, int(x)+self.resolution):
                        if (0<index<self.canvas_width):
                            self.canvas_trajectory[index] = y

                    # Draw
                    self.canvas.fill_style = 'black'
                    self.canvas.fill_rect(x, y, self.resolution, self.resolution)


    # Syncing data between 
    def on_mouse_up(self, x, y):
        """Stop editing data, pass canvas data to the backend/underlying data structure, redraw everything"""
        self.is_drawing = False
        self.trajectory_data: torch.Tensor = self.__tensor_map(self.__tensor_interpolate(self.canvas_trajectory,self.data_len),0, self.canvas_size, self.data_min,self.data_max)
        # print(self.trajectory_data)
        # print(self.canvas_trajectory.max(dim=0)[0])
        # print(self.canvas_trajectory)
        self.refresh()
        

    # Function to clear the canvas and reset the tensor
    def clear_canvas(self, event):
        """Wipe canvas"""
        self.canvas.clear()


    def refresh(self):
        """Call the function which has been assigned to this object, redraw interface"""
        self.refresh_function('Refreshing', self.latent_dimension, self.get_data())
        self.canvas.clear()
        self.draw_tensor()
        # plot tensor contents in canvas

    

    def get_data(self) -> torch.Tensor:
        """Get the underlying data currently represented in canvas"""
        return self.trajectory_data
