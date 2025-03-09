import numpy as np

class Node:
    """
    The Node class is used to store node information and keep track of the total number of 
    Degrees of Freedom (DOFs) of the problem. It introduces automatic bookkeeping in its 
    initialization, which efficiently keeps track of which DOFs belong to which nodes. This 
    makes it easier to assemble matrices from multiple elements.

    Attributes:
        x (float): The x-coordinate of the node.
        z (float): The z-coordinate of the node.
        p (numpy.array):  The load vector of the node.
        dofs (list): The Degrees of Freedom associated with the node.

    Methods:
        clear(): Clears the counting of degrees of freedom and number of nodes.
        __init__(x, z): The constructor for Node class.
        add_load(p): Adds the given loads to the node.
        get_coords(): Returns the coordinates of the node.
        __str__(): Returns a string representation of the node.
    """
    ndof = 0
    nn   = 0
    
    def clear():
        """
        Clears the counting of degrees of freedom and number of nodes.

        This method resets the class-level counters for degrees of freedom and number of nodes. 
        It should be used when you want to start a new problem from scratch.
        """
        Node.ndof = 0
        Node.nn = 0
        
    def __init__ (self, x, z): 
        """
        The constructor for Node class.

        Parameters:
            x (float):        The x-coordinate of the node.
            z (float):        The z-coordinate of the node.
            p (numpy.array):  The load vector of the node.
            dofs (list):      The Degrees of Freedom (u (in direction of x), w (in direction of z), phi (from z to x)) associated with the node.
        """

        self.x     = x
        self.z     = z
        self.p     = np.zeros(3)

        self.dofs  = [Node.ndof, Node.ndof+1, Node.ndof+2]

        Node.ndof += 3
        Node.nn   += 1

    def add_load (self, p):
        """
        Adds the given loads to the node.

        The load is a vector p, which includes the load in the x and y direction and a moment. 
        These loads are added to the existing loads of the node.

        Parameters:
            p (numpy.array): A vector containing the load in the x direction, the load in the y direction, 
                             and the moment. 
        """
        self.p += p

    def get_coords(self):
        """
        Returns the coordinates of the node.

        Returns:
           numpy.array: An array containing the x and z coordinates of the node.
        """
        return np.array([self.x, self.z])
    def __str__(self):
        """
        Returns a string representation of the node.

        Returns:
            str: A string representation of the node.
        """
        return f"This node has:\n - x coordinate={self.x},\n - z coordinate={self.z},\n - degrees of freedom={self.dofs},\n - load vector={self.p})"