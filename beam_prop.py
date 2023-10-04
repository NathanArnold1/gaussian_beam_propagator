import numpy as np
from numpy import sqrt, pi, matmul
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from abcd import waist_eq, waist2, thin_lens, s_mirror, prop, loopMat




# might try to reframe things as a Trie for easier access
class TrieNode:
     
    # Trie node class
    def __init__(self):
        self.children = []

 
        # isEndOfWord is True if node represent the end of the word
        self.isEnd = True
 
class Trie:
     
    # Trie data structure class
    def __init__(self):
        self.root = self.getNode()
 
    def getNode(self):
     
        # Returns new trie node (initialized to NULLs)
        return TrieNode()
 
    def _charToIndex(self,ch):
         
        # private helper function
        # Converts key current character into index
        # use only 'a' through 'z' and lower case
         
        return ord(ch)-ord('a')
 
 
    def insert(self,key):
         
        # If not present, inserts key into trie
        # If the key is prefix of trie node,
        # just marks leaf node
        pCrawl = self.root
        length = len(key)
        for level in range(length):
            index = self._charToIndex(key[level])
 
            # if current character is not present
            if not pCrawl.children[index]:
                pCrawl.children[index] = self.getNode()
            pCrawl = pCrawl.children[index]
 
        # mark last node as leaf
        pCrawl.isEndOfWord = True
 
    def search(self, key):
         
        # Search key in the trie
        # Returns true if key presents
        # in trie, else false
        pCrawl = self.root
        length = len(key)
        for level in range(length):
            index = self._charToIndex(key[level])
            if not pCrawl.children[index]:
                return False
            pCrawl = pCrawl.children[index]
 
        return pCrawl.isEndOfWord


# def give_sliders(plt, num_sliders):
#     ret = []
#     start_h = 0.05
#     spacing = 0.05
#     height = 0.03
#     start_w = 0.25
#     width = 0.4
#     for i in range(num_sliders):
#         ret.append(plt.axes([start_w, start_h + i*spacing, start_w + width, height], facecolor=axcolor))
#     return(ret)




 
class Plot:
    """
    To make things interactive it is helpful if we keep around the parameters used to make each plot.
    This is probably not the most efficient way to do it but for now it's fine.
    """
    def __init__(self, start, stop, w0, z0, λ, num_points, prog_ops, prog_params, color):
        self.start = start
        self.stop = stop
        self.w0 = w0
        self.z0 = z0
        self.λ = λ
        self.num_points = num_points
        self.prog_ops = prog_ops
        self.prog_params = prog_params
        self.color = color if not None else 'blue'
        self.plot, = plt.plot(self.x, self.y, lw=2, color=self.color)

    @property
    def x(self):
        return np.linspace(self.start, self.stop, self.num_points)
    
    @property
    def y(self):
        if not self.prog_ops:
            # print("calculating waist1")
            return waist_eq(self.λ, self.w0, self.x, self.z0)
        else:
            # print("calculating waist2")
            return waist2(self.λ, self.w0, self.z0, matmul(prop(self.x-self.start),loopMat(self.prog_ops, self.prog_params)))

    def set_new_data(self):
        self.plot.set_data(self.x, waist2(self.λ, self.w0, self.z0, matmul(prop(self.x-self.start),loopMat(self.prog_ops, self.prog_params))))

    def update_param(self, slider):
        def update(val):
            new_val = slider.val
            self.prog_params[-1] = new_val
            self.plot.set_data(self.x, waist2(self.λ, self.w0, self.z0, matmul(prop(self.x-self.start),loopMat(self.prog_ops, self.prog_params))))
            fig.canvas.draw_idle()
        
        slider.on_changed(update)

    def update_prop(self, val):
        pass


def make_plots(op_params, w0, z0,  start_buffer = 1, stop_buffer = 1, λ = 1550e-9, num_points = 100):
    """
    Given a list of operations and their relevant arguments, make plots that show a beam propagating through that system
    """
    ret = []
    num_ops = len(op_params)
    pos = 0
    progressive_ops = []
    progressive_params = []
    # plot starting buffer
    if start_buffer:
        p = Plot(pos - start_buffer, pos, w0, z0, λ, num_points, progressive_ops.copy(), progressive_params.copy(), 'blue')
        ret.append(p)
    for i in range(num_ops):
        op, param, color, variable = op_params[i]
        if op.__name__ == 'prop':
            # if it's a beam propagation step then plot, otherwise just add the operation to the list
            start = pos
            stop = pos + param
            pos += param
            p = Plot(start, stop, w0, z0, λ, num_points, progressive_ops.copy(), progressive_params.copy(), color)
            ret.append(p)
        progressive_ops.append(op)
        progressive_params.append(param)
    # Plot ending buffer
    if stop_buffer:
        start = pos
        stop = pos + stop_buffer
        p = Plot(start, stop, w0, z0, λ, num_points, progressive_ops, progressive_params, 'blue')
        ret.append(p)
    return ret





if __name__ == "__main__":
    s = 'static'
    v = 'variable'
    fig, ax = plt.subplots()

    curr_fig = plt.gcf()
    curr_fig.set_figheight(6)
    curr_fig.set_figwidth(10)
    plt.subplots_adjust(left=0.15, bottom=0.35)



    # Define the operations that will happen. Each sub-list item consists of an ABCD matrix, 
    # the relevant parameter for that matrix (e.g., focal length for a lens), a str
    # representing the color of the plot for that section, and a (currently) unused variable
    # for specifying static/dynamic parameters. Note that order of operations in the list is left-to-right,
    # so element 0 will happen first, then element 1, and so on.



    # w0 = 0.0005 # input waist size in meters
    # z0 = -0.001 # input location in meters (can't be zero because I get a div/0 error)
    # f1 = 1 # focal length of lens 1
    # d1 = 1 # progation distance 1
    # f2 = 4 # focal length of lens 2
    # op_params = [[prop, d1, 'green', s],[thin_lens, f1, None, s],[prop, d1, 'red', s],[thin_lens, f2, None, s],[prop, d1, 'green', s]]






    w0 = 5.2E-6 # input waist size in meters
    z0 = 0.0 # input location in meters (can't be zero because I get a div/0 error)
    d1 = 0.009595 # progation distance from fiber to collimating lens
    f1 = 9.6E-3 # focal length of lens 1
    d2 = 0.484214 # propagation distance from collimating lens to first spherical mirror
    roc1 = 2 # radius of curvature of spherical mirror 1
    d3 = 1.7444 # distance between spherical mirrors
    roc2 = 2 # radius of curvature of spherical mirror 2
    op_params = [[prop, d1, 'green', s],[thin_lens, f1, None, s],[prop, d2, 'red', s],[s_mirror, roc1, None, s],[prop, d3, 'green', s],[s_mirror, roc2, None, s],[prop, 1, 'green', s]]

    # Take operations and their parameters and make plots for each one
    plots = make_plots(op_params, w0, z0, start_buffer = 0.0000001)



    # from pprint import pprint
    # pprint(vars(plots[0]))

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)



    input_params = []
    input_sliders = []

    slider_start_h = 0.05
    slider_spacing = 0.05
    slider_height = 0.03
    slider_start_w = 0.25
    slider_width = 0.4

    # def find_variable_name(value, namespace):
    #     for name, val in namespace.items():
    #         if val is value:
    #             return name
    #     return None

    # The function to be called anytime a slider's value changes
    def update(val):
        # new_op_params = op_params.copy()

        curr_slider_values = []
        curr_prop_values = []
        ssp = []
        pos = 0

        for i in range(len(input_sliders)):
            curr_slider_values.append(input_sliders[i].val)

            # new_op_params[i][1] = input_sliders[i].val
            if op_params[i][0].__name__ == 'prop':
                start = pos
                stop = pos + input_sliders[i].val
                pos += input_sliders[i].val
                ssp.append((start,stop,pos))
                if i==0:
                    curr_prop_values.append([])
                else:
                    curr_prop_values.append(curr_slider_values[0:-1].copy())

        plots[-1].prog_params = curr_slider_values
        plots[-1].start = ssp[-1][2]
        plots[-1].stop = ssp[-1][2]+1
        plots[-1].set_new_data()

        for j in range(1,len(plots)-1):
            curr_plot = plots[j]

            curr_plot.start = ssp[j-1][0]
            curr_plot.stop = ssp[j-1][1]
            curr_plot.prog_params = curr_prop_values[j-1]
            curr_plot.set_new_data()



        # adjust the main plot to make room for the sliders
        # fig.subplots_adjust(left=0.25, bottom=0.25)
        fig.canvas.draw_idle()

    for i in range(len(op_params)):
        op, curr_param, color, variable = op_params[i]
            # variable_name = find_variable_name(curr_param, globals())

        # Make a horizontal slider to control the frequency.
        curr_ax = fig.add_axes([slider_start_w, slider_start_h + (len(op_params)-i-2)*slider_spacing, slider_start_w + slider_width, slider_height])
        curr_slider = Slider(
            ax=curr_ax,
            label='Parameter {}'.format(i+1),
            # label="{}".format(variable_name),
            valmin=curr_param*0.75,
            valmax=curr_param*1.25,
            valinit=curr_param,
        )

        curr_slider.on_changed(update)
        input_params.append(curr_param)
        input_sliders.append(curr_slider)

    # This is to make the plot interactive with sliders. Very hard to make this general
    plt.show()

