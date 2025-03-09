import numpy as np
import matplotlib.pyplot as plt
class Element_timo:
    """
    This Element class keeps track of each element in the model, including cross-section properties, 
    element orientation (for coordinate system transformations), and the nodes that make up each element. 
    With the help of the Node class, it also keeps track of which Degrees of Freedom (DOFs) belong to each element.

    This class is responsible for providing the element stiffness matrix in the global coordinate system 
    (for subsequent assembly) and postprocessing element-level fields. 

    This class describes an element combining extension and Timoshenko beam. A similar (or inherited) 
    class could also be implemented for different element types (e.g., shear beam, Euler-Bernoulli beam, cable elements, etc). 
    For simplicity, it is assumed that elements are all arranged in a 2D plane.

    Attributes:
        nodes (list): The two nodes of the element.
        EA (float): The axial stiffness of the element.
        EI (float): The flexural stiffness of the element.
        GA (float): The effective shear stiffness of the element
        beta (float): The ratio of shear to flexural stiffness

    Methods:
        clear(): Clears the counting of elements.
        __init__(self, nodes): Initializes an Element object.
        set_section(self, props): Sets the section properties of the element.
        global_dofs(self): Returns the global degrees of freedom associated with the element.
        stiffness(self): Calculate the stiffness matrix of the element.
        add_distributed_load(self, q): Adds a distributed load to the element.
        bending_moments(self, u_global, num_points=2): Calculate the bending moments along the element.
        full_displacement(self, u_global, num_points=2): Calculates the displacement along the element.
        plot_moment_diagram(self, u_elem, num_points=10, global_c=False, scale=1.0): Plots the bending moment diagram of the element.
        plot_displaced(self, u_elem, num_points=10, global_c=False, scale=1.0): Plots the displaced element.
        __str__(self): Returns a string representation of the Element object.
    """

    ne = 0

    def clear():
        """
        Clears the counting of elements

        This method resets the class-level counters for number of elements. 
        It should be used when you want to start a new problem from scratch.
        """
        Element.ne
    def __init__(self, node1, node2):
        """
        Initializes an Element object.

        Parameters:
        - node1 (Node): The first node of the element.
        - node2 (Node): The second node of the element.

        Attributes:
        - nodes (list): A list of Node objects representing the nodes of the element.
        - L (float): Length of the element.
        - cos (float): Cosine of the element's orientation angle.
        - sin (float): Sine of the element's orientation angle.
        - T (ndarray): Transformation matrix.
        - Tt (ndarray): Transpose of the transformation matrix.

        Returns:
        None
        """
        self.nodes = [node1, node2]

        self.L = np.sqrt((self.nodes[1].x - self.nodes[0].x)**2.0 + (self.nodes[1].z - self.nodes[0].z)**2.0)
        delta_x = self.nodes[1].x - self.nodes[0].x
        delta_z = self.nodes[1].z - self.nodes[0].z
        
        alpha = -np.arctan2(delta_z, delta_x)
        self.alpha_deg = alpha*180/np.pi
        T = np.zeros((6, 6))

        T[0, 0] = T[1, 1] = T[3, 3] = T[4, 4] = np.cos(alpha)
        T[0, 1] = T[3, 4] = -np.sin(alpha)
        T[1, 0] = T[4, 3] = np.sin(alpha)
        T[2, 2] = T[5, 5] = 1.0
        self.T = T
        self.Tt = np.transpose(T)

        self.q = np.array([0,0])
        
        Element.ne += 1
    
    def set_section(self, props):
        """
        Sets the section properties of the element.

        Parameters:
        - props (dict): A dictionary containing the section properties.
                        The dictionary should have the following keys:
                        - 'EA': The axial stiffness of the element.
                        - 'EI': The flexural stiffness of the element.
                        - 'GA': The shear stiffness of the element.

        Returns:
        None
        """
        if 'EA' in props:
            self.EA = props['EA']
        else:
            self.EA = 1.e20
        if 'EI' in props:
            self.EI = props['EI']
        else:
            self.EI = 1.e20
        if 'GA' in props:
            self.GA = props['GA']
        else:
            self.GA = 1.e20
    def global_dofs(self):
        """
        Returns the global degrees of freedom associated with the element.

        Returns:
            numpy.ndarray: Array containing the global degrees of freedom.
        """
        return np.hstack((self.nodes[0].dofs, self.nodes[1].dofs))
    def stiffness(self):
        """
        Calculate the stiffness matrix of the element.

        Returns:
        np.ndarray: The stiffness matrix of the element.
        """
        k = np.zeros((6, 6))

        EA = self.EA
        EI = self.EI
        GA = self.GA
        L = self.L
        beta = 12*EI/(GA*L**2)

        k[0, 0] = k[3, 3] = EA / L
        k[0, 3] = k[3, 0] = -EA / L
        k[1, 1] = k[4, 4] = 12 * EI /(L**3 * (beta+1))
        k[1, 4] = k[4, 1] = -12 * EI /(L**3 * (beta+1))
        k[1, 2] = k[2, 1] = k[1, 5] = k[5, 1] = -6 * EI /(L**2* (beta+1))
        k[2, 4] = k[4, 2] = k[4, 5] = k[5, 4] = 6 * EI /(L**2* (beta+1))
        k[2, 2] = k[5, 5] = (4+beta)*EI/L/(beta+1)
        k[2, 5] = k[5, 2] = (2-beta)*EI/L/(beta+1)


        return np.matmul(np.matmul(self.Tt, k), self.T)
    def add_distributed_load_sin(self, q):
        """
        Adds a sinusoidal distributed load q*sin(pi*x/L) to the element.

        Parameters:
            q (list): List of distributed load in x and z direction.

        Returns:
            None
        """

        l = self.L
        self.q = np.array(q)
        local_element_load =[l*q[0]/np.pi, l*q[1]/np.pi,-2*q[1]*l**2/np.pi**3,l*q[0]/np.pi, l*q[1]/np.pi,2*q[1]*l**2/np.pi**3]
        global_element_load = np.matmul(self.Tt, local_element_load)
        self.nodes[0].add_load(global_element_load[0:3])
        self.nodes[1].add_load(global_element_load[3:6])
         
    
    def bending_moments(self, u_global, num_points=2):
        """
        Calculate the bending moments along the element.

        Parameters:
        - u_global (numpy.ndarray): Global displacement vector.
        - num_points (int): Number of points to evaluate the bending moments. Default is 2.

        Returns:
        - M (numpy.ndarray): Array of bending moments at the specified points.
        """

        l = self.L
        q = self.q[1]
        EI = self.EI
        k = self.GA

        local_x = np.linspace(0.0, l, num_points)

        local_disp=np.matmul(self.T, u_global)
        beta=12*EI/(k*l**2)

        M = q*l**2/np.pi/np.pi*np.sin(np.pi*local_x/l) - 2*q*l**2/np.pi/np.pi/np.pi\
        +local_disp[1]*(6*EI/(1+beta)/l/l-12*EI*local_x/(1+beta)/l/l/l)\
        +local_disp[2]*(-(4+beta)*EI/(1+beta)/l+6*EI*local_x/(1+beta)/l/l)\
        +local_disp[4]*(-6*EI/(1+beta)/l/l+12*EI*local_x/(1+beta)/l/l/l)\
        +local_disp[5]*((beta-2)*EI/(1+beta)/l+6*EI*local_x/(1+beta)/l/l)

        self.mmt = M
        return M
    
    def full_displacement (self, u_global, num_points=2):
        """
        Calculates the displacement along the element.

        Args:
            u_global (numpy.ndarray): Global displacement vector of the element.
            num_points (int, optional): Number of points to calculate the bending moments. Default is 2.

        Returns:
            numpy.ndarray: Array of displacement along the element.
        """
        l = self.L
        q = self.q[1]
        EI = self.EI
        k = self.GA
        beta=12*EI/(k*l**2)

        local_x = np.linspace(0.0, l, num_points)

        local_disp=np.matmul(self.T, u_global)


        u = local_disp[3]*(local_x/l)+local_disp[0]*(1-local_x/l)

        w = q*l**2*local_x*(local_x-l)/np.pi**3/EI+(1+np.pi**2*EI/k/l**2)*(q*l**4/EI/np.pi**4)*np.sin(np.pi*local_x/l)\
            +local_disp[1] * (1 + (2*local_x**3/l**3-3*local_x**2/l**2-beta*local_x/l)/(1+beta))\
            +local_disp[2] * (-local_x**3/l**2+(1+beta/4)*2*local_x**2/l-(1+beta/2)*local_x)/(1+beta)\
            +local_disp[4] * (-2*local_x**3/l**3+3*local_x**2/l**2+beta*local_x/l)/(1+beta)\
            +local_disp[5] * (-local_x**3/l**2+(1/2-beta/4)*2*local_x**2/l+(beta/2)*local_x)/(1+beta)
        self.disp = w
        return u, w
    
    def plot_moment_diagram (self, u_elem, num_points=10, global_c=False, scale=1.0):
        """
        Plots the bending moment diagram of the element.

        Args:
            u_global (numpy.ndarray): Global displacement vector of the element.
            num_points (int, optional): Number of points to calculate the bending moments. Default is 2.
            global_c (bool, optional): If True, plots the bending moment diagram in the global coordinate system. Default is False (plots in local coordinate system).
            scale (float, optional): Scale factor for the bending moment diagram. Default is 1.0.

        Returns:
            None
        """
        import matplotlib.pyplot as plt

        x = np.linspace ( 0.0, self.L, num_points )
        M = self.bending_moments ( u_elem, num_points )
        xM_local = np.vstack((np.hstack([0,x,x[-1]]),np.hstack([0,M,0])*scale))
        if global_c:
            xM_global = np.matmul(self.Tt[0:2,:2],xM_local)
            xz_start_node = np.vstack((np.ones(num_points+2)*self.nodes[0].x, np.ones(num_points+2)*self.nodes[0].z))
            xz_Mlijn = xM_global + xz_start_node
            p = plt.plot(xz_Mlijn[0,:],xz_Mlijn[1,:])
            X0= self.nodes[0].x
            Z0= self.nodes[0].z
            X1= self.nodes[1].x
            Z1= self.nodes[1].z
            plt.plot((X0, X1), (Z0, Z1), color=p[0].get_color())
            plt.axis('off')
            plt.axis('equal')
        else:
            p = plt.plot(xM_local[0,:],xM_local[1,:])
            plt.xlabel ( "x" )
            plt.ylabel ( "M" )
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.title('Moment line')

    def plot_displaced(self, u_elem, num_points=10, global_c=False, scale=1.0):
        """
        Plots the displacd element.

        Args:
            u_global (numpy.ndarray): Global displacement vector of the element.
            num_points (int, optional): Number of points to calculate the bending moments. Default is 2.
            global_c (bool, optional): If True, plots the displacement diagram in the global coordinate system. Default is False (plots in local coordinate system).
            scale (float, optional): Scale factor for the displacement diagram. Default is 1.0.

        Returns:
            None
        """

        x = np.linspace ( 0.0, self.L, num_points )
        u, w = self.full_displacement ( u_elem, num_points )
        uw_local = np.vstack((x+u*scale,w*scale))
        if global_c:
            uw_global = np.matmul(self.Tt[:2,:2],uw_local)
            xz_start_node = np.vstack((np.ones(num_points)*self.nodes[0].x, np.ones(num_points)*self.nodes[0].z))
            uw = uw_global + xz_start_node
            p =  plt.plot(uw[0,:],uw[1,:])
            X0= self.nodes[0].x
            Z0= self.nodes[0].z
            X1= self.nodes[1].x
            Z1= self.nodes[1].z
            plt.plot((X0, X1), (Z0, Z1), color=p[0].get_color(),alpha=0.3)
            plt.axis('off')
            plt.axis('equal')
        else:
            p = plt.plot(uw_local[0,:],uw_local[1,:])
            plt.plot((0, self.L), (0, 0), color=p[0].get_color(),alpha=0.3)
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.title('Displaced structure')

    def __str__(self):
        """
        Returns a string representation of the Element object.
        
        The string includes the values of the node1, node2 attributes.
        """
        return f"Element connecting:\nnode #1:\n {self.nodes[0]}\nwith node #2:\n {self.nodes[1]}"
class Element:
    """
    The Element class keeps track of each element in the model, including cross-section properties, 
    element orientation (for coordinate system transformations), and the nodes that make up each element. 
    With the help of the Node class, it also keeps track of which Degrees of Freedom (DOFs) belong to each element.

    This class is responsible for providing the element stiffness matrix in the global coordinate system 
    (for subsequent assembly) and postprocessing element-level fields. 

    This class describes an element combining extension and Euler-Bernoulli bending. A similar (or inherited) 
    class could also be implemented for different element types (e.g., shear beam, Timoshenko beam, cable elements, etc). 
    For simplicity, it is assumed that elements are all arranged in a 2D plane.

    Attributes:
        nodes (list): The two nodes of the element.
        EA (float): The axial stiffness of the element.
        EI (float): The flexural stiffness of the element.

    Methods:
        clear(): Clears the counting of elements.
        __init__(self, nodes): Initializes an Element object.
        set_section(self, props): Sets the section properties of the element.
        global_dofs(self): Returns the global degrees of freedom associated with the element.
        stiffness(self): Calculate the stiffness matrix of the element.
        add_distributed_load(self, q): Adds a distributed load to the element.
        bending_moments(self, u_global, num_points=2): Calculate the bending moments along the element.
        full_displacement(self, u_global, num_points=2): Calculates the displacement along the element.
        plot_moment_diagram(self, u_elem, num_points=10, global_c=False, scale=1.0): Plots the bending moment diagram of the element.
        plot_displaced(self, u_elem, num_points=10, global_c=False, scale=1.0): Plots the displaced element.
        __str__(self): Returns a string representation of the Element object.
    """

    ne = 0

    def clear():
        """
        Clears the counting of elements

        This method resets the class-level counters for number of elements. 
        It should be used when you want to start a new problem from scratch.
        """
        Element.ne = 0
        
    def __init__(self, node1, node2):
        """
        Initializes an Element object.

        Parameters:
        - node1 (Node): The first node of the element.
        - node2 (Node): The second node of the element.

        Attributes:
        - nodes (list): A list of Node objects representing the nodes of the element.
        - L (float): Length of the element.
        - cos (float): Cosine of the element's orientation angle.
        - sin (float): Sine of the element's orientation angle.
        - T (ndarray): Transformation matrix.
        - Tt (ndarray): Transpose of the transformation matrix.

        Returns:
        None
        """
        self.nodes = [node1, node2]

        self.L = np.sqrt((self.nodes[1].x - self.nodes[0].x)**2.0 + (self.nodes[1].z - self.nodes[0].z)**2.0)
        delta_x = self.nodes[1].x - self.nodes[0].x
        delta_z = self.nodes[1].z - self.nodes[0].z
        
        alpha = -np.arctan2(delta_z, delta_x)
        self.alpha_deg = alpha*180/np.pi
        T = np.zeros((6, 6))

        T[0, 0] = T[1, 1] = T[3, 3] = T[4, 4] = np.cos(alpha)
        T[0, 1] = T[3, 4] = -np.sin(alpha)
        T[1, 0] = T[4, 3] = np.sin(alpha)
        T[2, 2] = T[5, 5] = 1.0
        self.T = T
        self.Tt = np.transpose(T)

        self.q = np.array([0,0])
        
        Element.ne += 1
    

    def set_section(self, props):
        """
        Sets the section properties of the element.

        Parameters:
        - props (dict): A dictionary containing the section properties.
                        The dictionary should have the following keys:
                        - 'EA': The axial stiffness of the element.
                        - 'EI': The flexural stiffness of the element.

        Returns:
        None
        """
        if 'EA' in props:
            self.EA = props['EA']
        else:
            self.EA = 1.e20
        if 'EI' in props:
            self.EI = props['EI']
        else:
            self.EI = 1.e20
    
    def global_dofs(self):
        """
        Returns the global degrees of freedom associated with the element.

        Returns:
            numpy.ndarray: Array containing the global degrees of freedom.
        """
        return np.hstack((self.nodes[0].dofs, self.nodes[1].dofs))

    def stiffness(self):
        """
        Calculate the stiffness matrix of the element.

        Returns:
        np.ndarray: The stiffness matrix of the element.
        """
        k = np.zeros((6, 6))

        EA = self.EA
        EI = self.EI
        L = self.L

        k[0, 0] = k[3, 3] = EA / L
        k[0, 3] = k[3, 0] = -EA / L
        k[1, 1] = k[4, 4] = 12 * EI / L**3
        k[1, 4] = k[4, 1] = -12 * EI / L**3
        k[1, 2] = k[2, 1] = k[1, 5] = k[5, 1] = -6 * EI / L**2
        k[2, 4] = k[4, 2] = k[4, 5] = k[5, 4] = 6 * EI / L**2
        k[2, 2] = k[5, 5] = 4 * EI / L
        k[2, 5] = k[5, 2] = 2 * EI / L

        return np.matmul(np.matmul(self.Tt, k), self.T)

    def add_distributed_load(self, q):
        """
        Adds a distributed load to the element.

        Parameters:
            q (list): List of distributed load in local x and z direction.

        Returns:
            None
        """

        l = self.L
        self.q = np.array(q)

        local_element_load =[0.5*l*q[0], 0.5*l*q[1],-q[1]*l**2/12, 0.5*l*q[0],0.5*l*q[1],q[1]*l**2/12]
        
        global_element_load = np.matmul(self.Tt, local_element_load)
    
        self.nodes[0].add_load(global_element_load[0:3])
        self.nodes[1].add_load(global_element_load[3:6])

    
    def bending_moments(self, u_global, num_points=2):
        """
        Calculate the bending moments along the element.

        Parameters:
        - u_global (numpy.ndarray): Global displacement vector.
        - num_points (int): Number of points to evaluate the bending moments. Default is 2.

        Returns:
        - M (numpy.ndarray): Array of bending moments at the specified points.
        """

        l = self.L
        q = self.q[1]
        EI = self.EI

        local_x = np.linspace(0.0, l, num_points)

        local_disp=np.matmul(self.T, u_global)
        
        phi1=local_disp[2]
        phi2=local_disp[5]
        w1=local_disp[1]
        w2=local_disp[4]

        m0 = q*(-l**2/12 + l*local_x/2 - local_x**2/2)
        m1 = 6*EI/l**2 - 12*EI*local_x/l**3
        m2 = -4*EI/l + 6*EI*local_x/l**2
        m3 = -6*EI/l**2 + 12*EI*local_x/l**3
        m4 = -2*EI/l + 6*EI*local_x/l**2
        
        
        M = m0 + m1*w1 + m2*phi1 + m3*w2 + m4*phi2

        self.mmt = M
        return M
    
    def full_displacement(self,u_global,num_points=2):
        """
        Calculates the displacement along the element.

        Args:
            u_global (numpy.ndarray): Global displacement vector of the element.
            num_points (int, optional): Number of points to calculate the bending moments. Default is 2.

        Returns:
            numpy.ndarray: Array of displacement along the element.
        """
        L = self.L
        qx= self.q[0]
        q = self.q[1]
        EI = self.EI
        EA = self.EA

        x = np.linspace(0.0, L, num_points)

        local_disp=np.matmul(self.T, u_global)
        # local_force =np.matmul(self.T, f_global)
        w1 = local_disp[1]
        w2 = local_disp[4]
        # f2 = local_force[2]
        # f4 = local_force[5]
        # # phi1 = (w2 - w1)/L + L/3/EI*f2 - L/6/EI*f4
        # # phi2 = (w2 - w1)/L + L/3/EI*f4 - L/6/EI*f2
        phi1 = local_disp[2]
        phi2 = local_disp[5]
        # print(phi1)
        # print(phi2)
        u = qx*(-L*x/(2*EA) + x**2/(2*EA))+local_disp[3]*(x/L)+local_disp[0]*(1-x/L)
        w = phi1*(-x + 2*x**2/L - x**3/L**2) + phi2*(x**2/L - x**3/L**2) + q*(L**2*x**2/(24*EI) - L*x**3/(12*EI) + x**4/(24*EI)) + w1*(1 - 3*x**2/L**2 + 2*x**3/L**3) + w2*(3*x**2/L**2 - 2*x**3/L**3)
        print(w)

        return u, w
    
    def plot_moment_diagram (self, u_elem, num_points=10, global_c=False, scale=1.0):
        """
        Plots the bending moment diagram of the element.

        Args:
            u_global (numpy.ndarray): Global displacement vector of the element.
            num_points (int, optional): Number of points to calculate the bending moments. Default is 2.
            global_c (bool, optional): If True, plots the bending moment diagram in the global coordinate system. Default is False (plots in local coordinate system).
            scale (float, optional): Scale factor for the bending moment diagram. Default is 1.0.

        Returns:
            None
        """
        import matplotlib.pyplot as plt

        x = np.linspace ( 0.0, self.L, num_points )
        M = self.bending_moments ( u_elem, num_points )
        xM_local = np.vstack((np.hstack([0,x,x[-1]]),np.hstack([0,M,0])*scale))
        if global_c:
            xM_global = np.matmul(self.Tt[0:2,:2],xM_local)
            xz_start_node = np.vstack((np.ones(num_points+2)*self.nodes[0].x, np.ones(num_points+2)*self.nodes[0].z))
            xz_Mlijn = xM_global + xz_start_node
            p = plt.plot(xz_Mlijn[0,:],xz_Mlijn[1,:])
            X0= self.nodes[0].x
            Z0= self.nodes[0].z
            X1= self.nodes[1].x
            Z1= self.nodes[1].z
            plt.plot((X0, X1), (Z0, Z1), color=p[0].get_color())
            plt.axis('off')
            plt.axis('equal')
        else:
            p = plt.plot(xM_local[0,:],xM_local[1,:])
            plt.xlabel ( "x" )
            plt.ylabel ( "M" )
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.title('Moment line')

    def plot_displaced(self, u_elem, num_points=10, global_c=False, scale=1.0):
        """
        Plots the displacd element.

        Args:
            u_global (numpy.ndarray): Global displacement vector of the element.
            num_points (int, optional): Number of points to calculate the bending moments. Default is 2.
            global_c (bool, optional): If True, plots the displacement diagram in the global coordinate system. Default is False (plots in local coordinate system).
            scale (float, optional): Scale factor for the displacement diagram. Default is 1.0.

        Returns:
            None
        """

        x = np.linspace ( 0.0, self.L, num_points)
        u, w = self.full_displacement (u_elem,num_points)
        uw_local = np.vstack((x+u*scale,w*scale))
        if global_c:
            uw_global = np.matmul(self.Tt[:2,:2],uw_local)
            xz_start_node = np.vstack((np.ones(num_points)*self.nodes[0].x, np.ones(num_points)*self.nodes[0].z))
            uw = uw_global + xz_start_node
            p =  plt.plot(uw[0,:],uw[1,:])
            X0= self.nodes[0].x
            Z0= self.nodes[0].z
            X1= self.nodes[1].x
            Z1= self.nodes[1].z
            plt.plot((X0, X1), (Z0, Z1), color=p[0].get_color(),alpha=0.3)
            plt.axis('off')
            plt.axis('equal')
        else:
            p = plt.plot(uw_local[0,:],uw_local[1,:])
            plt.plot((0, self.L), (0, 0), color=p[0].get_color(),alpha=0.3)
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.title('Displaced structure')

    def __str__(self):
        """
        Returns a string representation of the Element object.
        
        The string includes the values of the node1, node2 attributes.
        """
        return f"Element connecting:\nnode #1:\n {self.nodes[0]}\nwith node #2:\n {self.nodes[1]}"
class Element_cable:
    """
    This Element class keeps track of each element in the model, including cross-section properties, 
    element orientation (for coordinate system transformations), and the nodes that make up each element. 
    With the help of the Node class, it also keeps track of which Degrees of Freedom (DOFs) belong to each element.

    This class is responsible for providing the element stiffness matrix in the global coordinate system 
    (for subsequent assembly) and postprocessing element-level fields. 

    This class describes an element combining extension and Timoshenko beam. A similar (or inherited) 
    class could also be implemented for different element types (e.g., shear beam, Euler-Bernoulli beam, cable elements, etc). 
    For simplicity, it is assumed that elements are all arranged in a 2D plane.

    Attributes:
        nodes (list): The two nodes of the element.
        EA (float): The axial stiffness of the element.
        EI (float): The flexural stiffness of the element.
        GA (float): The effective shear stiffness of the element
        beta (float): The ratio of shear to flexural stiffness

    Methods:
        clear(): Clears the counting of elements.
        __init__(self, nodes): Initializes an Element object.
        set_section(self, props): Sets the section properties of the element.
        global_dofs(self): Returns the global degrees of freedom associated with the element.
        stiffness(self): Calculate the stiffness matrix of the element.
        add_distributed_load(self, q): Adds a distributed load to the element.
        bending_moments(self, u_global, num_points=2): Calculate the bending moments along the element.
        full_displacement(self, u_global, num_points=2): Calculates the displacement along the element.
        plot_moment_diagram(self, u_elem, num_points=10, global_c=False, scale=1.0): Plots the bending moment diagram of the element.
        plot_displaced(self, u_elem, num_points=10, global_c=False, scale=1.0): Plots the displaced element.
        __str__(self): Returns a string representation of the Element object.
    """

    ne = 0

    def clear():
        """
        Clears the counting of elements

        This method resets the class-level counters for number of elements. 
        It should be used when you want to start a new problem from scratch.
        """
        Element.ne
    def __init__(self, node1, node2):
        """
        Initializes an Element object.

        Parameters:
        - node1 (Node): The first node of the element.
        - node2 (Node): The second node of the element.

        Attributes:
        - nodes (list): A list of Node objects representing the nodes of the element.
        - L (float): Length of the element.
        - cos (float): Cosine of the element's orientation angle.
        - sin (float): Sine of the element's orientation angle.
        - T (ndarray): Transformation matrix.
        - Tt (ndarray): Transpose of the transformation matrix.

        Returns:
        None
        """
        self.nodes = [node1, node2]

        self.L = np.sqrt((self.nodes[1].x - self.nodes[0].x)**2.0 + (self.nodes[1].z - self.nodes[0].z)**2.0)
        delta_x = self.nodes[1].x - self.nodes[0].x
        delta_z = self.nodes[1].z - self.nodes[0].z
        
        alpha = -np.arctan2(delta_z, delta_x)
        self.alpha_deg = alpha*180/np.pi
        T = np.zeros((6, 6))

        T[0, 0] = T[1, 1] = T[3, 3] = T[4, 4] = np.cos(alpha)
        T[0, 1] = T[3, 4] = -np.sin(alpha)
        T[1, 0] = T[4, 3] = np.sin(alpha)
        T[2, 2] = T[5, 5] = 1.0
        self.T = T
        self.Tt = np.transpose(T)

        self.q = np.array([0,0])
        
        Element.ne += 1
    
    def set_section(self, props):
        """
        Sets the section properties of the element.

        Parameters:
        - props (dict): A dictionary containing the section properties.
                        The dictionary should have the following keys:
                        - 'EA': The axial stiffness of the element.
                        - 'EI': The flexural stiffness of the element.
                        - 'GA': The shear stiffness of the element.

        Returns:
        None
        """
        if 'EA' in props:
            self.EA = props['EA']
        else:
            self.EA = 1.e20
        if 'EI' in props:
            self.EI = props['EI']
        else:
            self.EI = 1.e20
        if 'GA' in props:
            self.GA = props['GA']
        else:
            self.GA = 1.e20
    def global_dofs(self):
        """
        Returns the global degrees of freedom associated with the element.

        Returns:
            numpy.ndarray: Array containing the global degrees of freedom.
        """
        return np.hstack((self.nodes[0].dofs, self.nodes[1].dofs))
    def stiffness(self):
        """
        Calculate the stiffness matrix of the element.

        Returns:
        np.ndarray: The stiffness matrix of the element.
        """
        k = np.zeros((6, 6))

        EA = self.EA
        EI = 0
        GA = 0
        L = self.L
        beta = 12*EI/(GA*L**2)

        k[0, 0] = k[3, 3] = EA / L
        k[0, 3] = k[3, 0] = -EA / L
        k[1, 1] = k[4, 4] = 12 * EI /(L**3 * (beta+1))
        k[1, 4] = k[4, 1] = -12 * EI /(L**3 * (beta+1))
        k[1, 2] = k[2, 1] = k[1, 5] = k[5, 1] = -6 * EI /(L**2* (beta+1))
        k[2, 4] = k[4, 2] = k[4, 5] = k[5, 4] = 6 * EI /(L**2* (beta+1))
        k[2, 2] = k[5, 5] = (4+beta)*EI/L/(beta+1)
        k[2, 5] = k[5, 2] = (2-beta)*EI/L/(beta+1)


        return np.matmul(np.matmul(self.Tt, k), self.T)
    
    def add_distributed_load_sin(self, q):
        """
        Adds a sinusoidal distributed load q*sin(pi*x/L) to the element.

        Parameters:
            q (list): List of distributed load in x and z direction.

        Returns:
            None
        """

        l = self.L
        self.q = np.array(q)
        local_element_load =[l*q[0]/np.pi, l*q[1]/np.pi,-2*q[1]*l**2/np.pi**3,l*q[0]/np.pi, l*q[1]/np.pi,2*q[1]*l**2/np.pi**3]
        global_element_load = np.matmul(self.Tt, local_element_load)
        self.nodes[0].add_load(global_element_load[0:3])
        self.nodes[1].add_load(global_element_load[3:6])
         
    
    def bending_moments(self, u_global, num_points=2):
        """
        Calculate the bending moments along the element.

        Parameters:
        - u_global (numpy.ndarray): Global displacement vector.
        - num_points (int): Number of points to evaluate the bending moments. Default is 2.

        Returns:
        - M (numpy.ndarray): Array of bending moments at the specified points.
        """

        l = self.L
        q = self.q[1]
        EI = 0
        k = 0

        local_x = np.linspace(0.0, l, num_points)

        local_disp=np.matmul(self.T, u_global)
        beta=12*EI/(k*l**2)

        M = q*l**2/np.pi/np.pi*np.sin(np.pi*local_x/l) - 2*q*l**2/np.pi/np.pi/np.pi\
        +local_disp[1]*(6*EI/(1+beta)/l/l-12*EI*local_x/(1+beta)/l/l/l)\
        +local_disp[2]*(-(4+beta)*EI/(1+beta)/l+6*EI*local_x/(1+beta)/l/l)\
        +local_disp[4]*(-6*EI/(1+beta)/l/l+12*EI*local_x/(1+beta)/l/l/l)\
        +local_disp[5]*((beta-2)*EI/(1+beta)/l+6*EI*local_x/(1+beta)/l/l)

        M = 0

        self.mmt = M
        return M
    
    def full_displacement (self, u_global, num_points=2):
        """
        Calculates the displacement along the element.

        Args:
            u_global (numpy.ndarray): Global displacement vector of the element.
            num_points (int, optional): Number of points to calculate the bending moments. Default is 2.

        Returns:
            numpy.ndarray: Array of displacement along the element.
        """
        l = self.L
        q = self.q[0]
        EI = 0
        k = 0
        beta=12*EI/(k*l**2)

        local_x = np.linspace(0.0, l, num_points)

        local_disp=np.matmul(self.T, u_global)


        u = local_disp[3]*(local_x/l)+local_disp[0]*(1-local_x/l)

        w = q*l**2*local_x*(local_x-l)/np.pi**3/EI+(1+np.pi**2*EI/k/l**2)*(q*l**4/EI/np.pi**4)*np.sin(np.pi*local_x/l)\
            +local_disp[1] * (1 + (2*local_x**3/l**3-3*local_x**2/l**2-beta*local_x/l)/(1+beta))\
            +local_disp[2] * (-local_x**3/l**2+(1+beta/4)*2*local_x**2/l-(1+beta/2)*local_x)/(1+beta)\
            +local_disp[4] * (-2*local_x**3/l**3+3*local_x**2/l**2+beta*local_x/l)/(1+beta)\
            +local_disp[5] * (-local_x**3/l**2+(1/2-beta/4)*2*local_x**2/l+(beta/2)*local_x)/(1+beta)
        self.disp = w
        return u, w
    
    def plot_moment_diagram (self, u_elem, num_points=10, global_c=False, scale=1.0):
        """
        Plots the bending moment diagram of the element.

        Args:
            u_global (numpy.ndarray): Global displacement vector of the element.
            num_points (int, optional): Number of points to calculate the bending moments. Default is 2.
            global_c (bool, optional): If True, plots the bending moment diagram in the global coordinate system. Default is False (plots in local coordinate system).
            scale (float, optional): Scale factor for the bending moment diagram. Default is 1.0.

        Returns:
            None
        """
        import matplotlib.pyplot as plt

        x = np.linspace ( 0.0, self.L, num_points )
        M = self.bending_moments ( u_elem, num_points )
        xM_local = np.vstack((np.hstack([0,x,x[-1]]),np.hstack([0,M,0])*scale))
        if global_c:
            xM_global = np.matmul(self.Tt[0:2,:2],xM_local)
            xz_start_node = np.vstack((np.ones(num_points+2)*self.nodes[0].x, np.ones(num_points+2)*self.nodes[0].z))
            xz_Mlijn = xM_global + xz_start_node
            p = plt.plot(xz_Mlijn[0,:],xz_Mlijn[1,:])
            X0= self.nodes[0].x
            Z0= self.nodes[0].z
            X1= self.nodes[1].x
            Z1= self.nodes[1].z
            plt.plot((X0, X1), (Z0, Z1), color=p[0].get_color())
            plt.axis('off')
            plt.axis('equal')
        else:
            p = plt.plot(xM_local[0,:],xM_local[1,:])
            plt.xlabel ( "x" )
            plt.ylabel ( "M" )
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.title('Moment line')

    def plot_displaced(self, u_elem, num_points=10, global_c=False, scale=1.0):
        """
        Plots the displacd element.

        Args:
            u_global (numpy.ndarray): Global displacement vector of the element.
            num_points (int, optional): Number of points to calculate the bending moments. Default is 2.
            global_c (bool, optional): If True, plots the displacement diagram in the global coordinate system. Default is False (plots in local coordinate system).
            scale (float, optional): Scale factor for the displacement diagram. Default is 1.0.

        Returns:
            None
        """

        x = np.linspace ( 0.0, self.L, num_points )
        u, w = self.full_displacement ( u_elem, num_points )
        uw_local = np.vstack((x+u*scale,w*scale))
        if global_c:
            uw_global = np.matmul(self.Tt[:2,:2],uw_local)
            xz_start_node = np.vstack((np.ones(num_points)*self.nodes[0].x, np.ones(num_points)*self.nodes[0].z))
            uw = uw_global + xz_start_node
            p =  plt.plot(uw[0,:],uw[1,:])
            X0= self.nodes[0].x
            Z0= self.nodes[0].z
            X1= self.nodes[1].x
            Z1= self.nodes[1].z
            plt.plot((X0, X1), (Z0, Z1), color=p[0].get_color(),alpha=0.3)
            plt.axis('off')
            plt.axis('equal')
        else:
            p = plt.plot(uw_local[0,:],uw_local[1,:])
            plt.plot((0, self.L), (0, 0), color=p[0].get_color(),alpha=0.3)
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.title('Displaced structure')

    def __str__(self):
        """
        Returns a string representation of the Element object.
        
        The string includes the values of the node1, node2 attributes.
        """
        return f"Element connecting:\nnode #1:\n {self.nodes[0]}\nwith node #2:\n {self.nodes[1]}"
class Element_truss:
    """
    The Element class keeps track of each element in the model, including cross-section properties, 
    element orientation (for coordinate system transformations), and the nodes that make up each element. 
    With the help of the Node class, it also keeps track of which Degrees of Freedom (DOFs) belong to each element.

    This class is responsible for providing the element stiffness matrix in the global coordinate system 
    (for subsequent assembly) and postprocessing element-level fields. 

    This class describes an element combining extension and Euler-Bernoulli bending with two hinges. A similar (or inherited) 
    class could also be implemented for different element types (e.g., shear beam, Timoshenko beam, cable elements, etc). 
    For simplicity, it is assumed that elements are all arranged in a 2D plane.

    Attributes:
        nodes (list): The two nodes of the element.
        EA (float): The axial stiffness of the element.
        EI (float): The flexural stiffness of the element.

    Methods:
        clear(): Clears the counting of elements.
        __init__(self, nodes): Initializes an Element object.
        set_section(self, props): Sets the section properties of the element.
        global_dofs(self): Returns the global degrees of freedom associated with the element.
        stiffness(self): Calculate the stiffness matrix of the element.
        add_distributed_load(self, q): Adds a distributed load to the element.
        bending_moments(self, u_global, num_points=2): Calculate the bending moments along the element.
        full_displacement(self, u_global, num_points=2): Calculates the displacement along the element.
        plot_moment_diagram(self, u_elem, num_points=10, global_c=False, scale=1.0): Plots the bending moment diagram of the element.
        plot_displaced(self, u_elem, num_points=10, global_c=False, scale=1.0): Plots the displaced element.
        __str__(self): Returns a string representation of the Element object.
    """

    ne = 0

    def clear():
        """
        Clears the counting of elements

        This method resets the class-level counters for number of elements. 
        It should be used when you want to start a new problem from scratch.
        """
        Element.ne = 0
        
    def __init__(self, node1, node2):
        """
        Initializes an Element object.

        Parameters:
        - node1 (Node): The first node of the element.
        - node2 (Node): The second node of the element.

        Attributes:
        - nodes (list): A list of Node objects representing the nodes of the element.
        - L (float): Length of the element.
        - cos (float): Cosine of the element's orientation angle.
        - sin (float): Sine of the element's orientation angle.
        - T (ndarray): Transformation matrix.
        - Tt (ndarray): Transpose of the transformation matrix.

        Returns:
        None
        """
        self.nodes = [node1, node2]

        self.L = np.sqrt((self.nodes[1].x - self.nodes[0].x)**2.0 + (self.nodes[1].z - self.nodes[0].z)**2.0)
        delta_x = self.nodes[1].x - self.nodes[0].x
        delta_z = self.nodes[1].z - self.nodes[0].z
        
        alpha = -np.arctan2(delta_z, delta_x)
        self.alpha_deg = alpha*180/np.pi
        T = np.zeros((6, 6))

        T[0, 0] = T[1, 1] = T[3, 3] = T[4, 4] = np.cos(alpha)
        T[0, 1] = T[3, 4] = -np.sin(alpha)
        T[1, 0] = T[4, 3] = np.sin(alpha)
        T[2, 2] = T[5, 5] = 1.0
        self.T = T
        self.Tt = np.transpose(T)

        self.q = np.array([0,0])
        
        Element.ne += 1
    

    def set_section(self, props):
        """
        Sets the section properties of the element.

        Parameters:
        - props (dict): A dictionary containing the section properties.
                        The dictionary should have the following keys:
                        - 'EA': The axial stiffness of the element.
                        - 'EI': The flexural stiffness of the element.

        Returns:
        None
        """
        if 'EA' in props:
            self.EA = props['EA']
        else:
            self.EA = 1.e20
        if 'EI' in props:
            self.EI = props['EI']
        else:
            self.EI = 1.e20
    
    def global_dofs(self):
        """
        Returns the global degrees of freedom associated with the element.

        Returns:
            numpy.ndarray: Array containing the global degrees of freedom.
        """
        return np.hstack((self.nodes[0].dofs, self.nodes[1].dofs))

    def stiffness(self):
        """
        Calculate the stiffness matrix of the element.

        Returns:
        np.ndarray: The stiffness matrix of the element.
        """
        k = np.zeros((6, 6))

        EA = self.EA
        EI = self.EI
        L = self.L

        k[0, 0] = k[3, 3] = EA / L
        k[0, 3] = k[3, 0] = -EA / L


        return np.matmul(np.matmul(self.Tt, k), self.T)

    # def add_distributed_load(self, q):
    #     """
    #     Adds a distributed load to the element.

    #     Parameters:
    #         q (list): List of distributed load in local x and z direction.

    #     Returns:
    #         None
    #     """

    #     l = self.L
    #     self.q = np.array(q)

    #     local_element_load =[0.5*l*q[0], 0.5*l*q[1],-q[1]*l**2/12, 0.5*l*q[0],0.5*l*q[1],q[1]*l**2/12]
        
    #     global_element_load = np.matmul(self.Tt, local_element_load)
    
    #     self.nodes[0].add_load(global_element_load[0:3])
    #     self.nodes[1].add_load(global_element_load[3:6])

    
    def bending_moments(self, u_global, num_points=2):
        """
        Calculate the bending moments along the element.

        Parameters:
        - u_global (numpy.ndarray): Global displacement vector.
        - num_points (int): Number of points to evaluate the bending moments. Default is 2.

        Returns:
        - M (numpy.ndarray): Array of bending moments at the specified points.
        """

        l = self.L
        q = self.q[1]
        EI = self.EI

        local_x = np.linspace(0.0, l, num_points)

        local_disp=np.matmul(self.T, u_global)
        
        phi1=local_disp[2]
        phi2=local_disp[5]
        w1=local_disp[1]
        w2=local_disp[4]

        m0 = q*(-l**2/12 + l*local_x/2 - local_x**2/2)
        m1 = 6*EI/l**2 - 12*EI*local_x/l**3
        m2 = -4*EI/l + 6*EI*local_x/l**2
        m3 = -6*EI/l**2 + 12*EI*local_x/l**3
        m4 = -2*EI/l + 6*EI*local_x/l**2
        
        
        M = m0 + m1*w1 + m2*phi1 + m3*w2 + m4*phi2
        
        return np.zeros_like(local_x)
    
    def full_displacement (self, u_global, num_points=2):
        """
        Calculates the displacement along the element.

        Args:
            u_global (numpy.ndarray): Global displacement vector of the element.
            num_points (int, optional): Number of points to calculate the bending moments. Default is 2.

        Returns:
            numpy.ndarray: Array of displacement along the element.
        """
        L = self.L
        qx= self.q[0]
        q = self.q[1]
        EI = self.EI
        EA = self.EA

        x = np.linspace(0.0, L, num_points)

        local_disp=np.matmul(self.T, u_global)
        w1 = local_disp[1]
        phi1= 0
        w2 = local_disp[4]
        phi2 = 0

        u = qx*(-L*x/(2*EA) + x**2/(2*EA))+local_disp[3]*(x/L)+local_disp[0]*(1-x/L)

        # w = phi1*(-x + 2*x**2/L - x**3/L**2) + phi2*(x**2/L - x**3/L**2) + q*(L**2*x**2/(24*EI) - L*x**3/(12*EI) + x**4/(24*EI)) + w1*(1 - 3*x**2/L**2 + 2*x**3/L**3) + w2*(3*x**2/L**2 - 2*x**3/L**3)
        w = w1 + (w2-w1)*x/L

        return u, w
    
    def plot_moment_diagram (self, u_elem, num_points=10, global_c=False, scale=1.0):
        """
        Plots the bending moment diagram of the element.

        Args:
            u_global (numpy.ndarray): Global displacement vector of the element.
            num_points (int, optional): Number of points to calculate the bending moments. Default is 2.
            global_c (bool, optional): If True, plots the bending moment diagram in the global coordinate system. Default is False (plots in local coordinate system).
            scale (float, optional): Scale factor for the bending moment diagram. Default is 1.0.

        Returns:
            None
        """
        import matplotlib.pyplot as plt

        x = np.linspace ( 0.0, self.L, num_points )
        M = self.bending_moments ( u_elem, num_points )
        xM_local = np.vstack((np.hstack([0,x,x[-1]]),np.hstack([0,M,0])*scale))
        if global_c:
            xM_global = np.matmul(self.Tt[0:2,:2],xM_local)
            xz_start_node = np.vstack((np.ones(num_points+2)*self.nodes[0].x, np.ones(num_points+2)*self.nodes[0].z))
            xz_Mlijn = xM_global + xz_start_node
            p = plt.plot(xz_Mlijn[0,:],xz_Mlijn[1,:])
            X0= self.nodes[0].x
            Z0= self.nodes[0].z
            X1= self.nodes[1].x
            Z1= self.nodes[1].z
            plt.plot((X0, X1), (Z0, Z1), color=p[0].get_color())
            plt.axis('off')
            plt.axis('equal')
        else:
            p = plt.plot(xM_local[0,:],xM_local[1,:])
            plt.xlabel ( "x" )
            plt.ylabel ( "M" )
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.title('Moment line')

    def plot_displaced(self, u_elem, num_points=10, global_c=False, scale=1.0):
        """
        Plots the displacd element.

        Args:
            u_global (numpy.ndarray): Global displacement vector of the element.
            num_points (int, optional): Number of points to calculate the bending moments. Default is 2.
            global_c (bool, optional): If True, plots the displacement diagram in the global coordinate system. Default is False (plots in local coordinate system).
            scale (float, optional): Scale factor for the displacement diagram. Default is 1.0.

        Returns:
            None
        """

        x = np.linspace ( 0.0, self.L, num_points )
        u, w = self.full_displacement ( u_elem, num_points )
        uw_local = np.vstack((x+u*scale,w*scale))
        if global_c:
            uw_global = np.matmul(self.Tt[:2,:2],uw_local)
            xz_start_node = np.vstack((np.ones(num_points)*self.nodes[0].x, np.ones(num_points)*self.nodes[0].z))
            uw = uw_global + xz_start_node
            p =  plt.plot(uw[0,:],uw[1,:])
            X0= self.nodes[0].x
            Z0= self.nodes[0].z
            X1= self.nodes[1].x
            Z1= self.nodes[1].z
            plt.plot((X0, X1), (Z0, Z1), color=p[0].get_color(),alpha=0.3)
            plt.axis('off')
            plt.axis('equal')
        else:
            p = plt.plot(uw_local[0,:],uw_local[1,:])
            plt.plot((0, self.L), (0, 0), color=p[0].get_color(),alpha=0.3)
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.title('Displaced structure')

    def __str__(self):
        """
        Returns a string representation of the Element object.
        
        The string includes the values of the node1, node2 attributes.
        """
        return f"Element connecting:\nnode #1:\n {self.nodes[0]}\nwith node #2:\n {self.nodes[1]}"
class Element_onehinge:
    """
    The Element class keeps track of each element in the model, including cross-section properties, 
    element orientation (for coordinate system transformations), and the nodes that make up each element. 
    With the help of the Node class, it also keeps track of which Degrees of Freedom (DOFs) belong to each element.

    This class is responsible for providing the element stiffness matrix in the global coordinate system 
    (for subsequent assembly) and postprocessing element-level fields. 

    This class describes an element combining extension and Euler-Bernoulli bending with two hinges. A similar (or inherited) 
    class could also be implemented for different element types (e.g., shear beam, Timoshenko beam, cable elements, etc). 
    For simplicity, it is assumed that elements are all arranged in a 2D plane.

    Attributes:
        nodes (list): The two nodes of the element.
        EA (float): The axial stiffness of the element.
        EI (float): The flexural stiffness of the element.

    Methods:
        clear(): Clears the counting of elements.
        __init__(self, nodes): Initializes an Element object.
        set_section(self, props): Sets the section properties of the element.
        global_dofs(self): Returns the global degrees of freedom associated with the element.
        stiffness(self): Calculate the stiffness matrix of the element.
        add_distributed_load(self, q): Adds a distributed load to the element.
        bending_moments(self, u_global, num_points=2): Calculate the bending moments along the element.
        full_displacement(self, u_global, num_points=2): Calculates the displacement along the element.
        plot_moment_diagram(self, u_elem, num_points=10, global_c=False, scale=1.0): Plots the bending moment diagram of the element.
        plot_displaced(self, u_elem, num_points=10, global_c=False, scale=1.0): Plots the displaced element.
        __str__(self): Returns a string representation of the Element object.
    """

    ne = 0

    def clear():
        """
        Clears the counting of elements

        This method resets the class-level counters for number of elements. 
        It should be used when you want to start a new problem from scratch.
        """
        Element.ne = 0
        
    def __init__(self, node1, node2):
        """
        Initializes an Element object.

        Parameters:
        - node1 (Node): The first node of the element.
        - node2 (Node): The second node of the element.

        Attributes:
        - nodes (list): A list of Node objects representing the nodes of the element.
        - L (float): Length of the element.
        - cos (float): Cosine of the element's orientation angle.
        - sin (float): Sine of the element's orientation angle.
        - T (ndarray): Transformation matrix.
        - Tt (ndarray): Transpose of the transformation matrix.

        Returns:
        None
        """
        self.nodes = [node1, node2]

        self.L = np.sqrt((self.nodes[1].x - self.nodes[0].x)**2.0 + (self.nodes[1].z - self.nodes[0].z)**2.0)
        delta_x = self.nodes[1].x - self.nodes[0].x
        delta_z = self.nodes[1].z - self.nodes[0].z
        
        alpha = -np.arctan2(delta_z, delta_x)
        self.alpha_deg = alpha*180/np.pi
        T = np.zeros((6, 6))

        T[0, 0] = T[1, 1] = T[3, 3] = T[4, 4] = np.cos(alpha)
        T[0, 1] = T[3, 4] = -np.sin(alpha)
        T[1, 0] = T[4, 3] = np.sin(alpha)
        T[2, 2] = T[5, 5] = 1.0
        self.T = T
        self.Tt = np.transpose(T)

        self.q = np.array([0,0])
        
        Element.ne += 1
    

    def set_section(self, props):
        """
        Sets the section properties of the element.

        Parameters:
        - props (dict): A dictionary containing the section properties.
                        The dictionary should have the following keys:
                        - 'EA': The axial stiffness of the element.
                        - 'EI': The flexural stiffness of the element.

        Returns:
        None
        """
        if 'EA' in props:
            self.EA = props['EA']
        else:
            self.EA = 1.e20
        if 'EI' in props:
            self.EI = props['EI']
        else:
            self.EI = 1.e20
    
    def global_dofs(self):
        """
        Returns the global degrees of freedom associated with the element.

        Returns:
            numpy.ndarray: Array containing the global degrees of freedom.
        """
        return np.hstack((self.nodes[0].dofs, self.nodes[1].dofs))

    def stiffness(self):
        """
        Calculate the stiffness matrix of the element.

        Returns:
        np.ndarray: The stiffness matrix of the element.
        """
        k = np.zeros((6, 6))

        EA = self.EA
        EI = self.EI
        L = self.L

        k[0, 0] = k[3, 3] = EA / L
        k[0, 3] = k[3, 0] = -EA / L
        k[1, 1] = k[4, 4] = 3 * EI / L**3
        k[1, 2] = k[2, 1] = -3 * EI / L**2
        k[2, 4] = k[4, 2] = 3 *EI / L**2
        k[2, 2] = 3* EI / L
        k[1, 4] = k[4, 1] = -3 * EI / L**3
        return np.matmul(np.matmul(self.Tt, k), self.T)

    def add_distributed_load(self, q):
        """
        Adds a distributed load to the element.

        Parameters:
            q (list): List of distributed load in local x and z direction.

        Returns:
            None
        """

        l = self.L
        self.q = np.array(q)

        local_element_load =[0.5*l*q[0], 5*l*q[1]/8,-q[1]*l**2/8, 0.5*l*q[0],3*l*q[1]/8,0]
        
        global_element_load = np.matmul(self.Tt, local_element_load)
    
        self.nodes[0].add_load(global_element_load[0:3])
        self.nodes[1].add_load(global_element_load[3:6])

    
    def bending_moments(self, u_global, num_points=2):
        """
        Calculate the bending moments along the element.

        Parameters:
        - u_global (numpy.ndarray): Global displacement vector.
        - num_points (int): Number of points to evaluate the bending moments. Default is 2.

        Returns:
        - M (numpy.ndarray): Array of bending moments at the specified points.
        """

        L = self.L
        q = self.q[1]
        EI = self.EI

        local_x = np.linspace(0.0, L, num_points)
        x = local_x
        local_disp=np.matmul(self.T, u_global)
        
        phi_1=local_disp[2]
        w_1=local_disp[1]
        w_2=local_disp[4]
        
    
        M = (-4*L**3*q*x**2 + L*(-24*EI*L*phi_1 + 24*EI*w_1 - 24*EI*w_2 - L**4*q) + x*(24*EI*L*phi_1 - 24*EI*w_1 + 24*EI*w_2 + 5*L**4*q))/(8*L**3)

        self.mmt = M
        return M
    
    def full_displacement (self, u_global, num_points=2):
        """
        Calculates the displacement along the element.

        Args:
            u_global (numpy.ndarray): Global displacement vector of the element.
            num_points (int, optional): Number of points to calculate the bending moments. Default is 2.

        Returns:
            numpy.ndarray: Array of displacement along the element.
        """
        L = self.L
        qx= self.q[0]
        q = self.q[1]
        EI = self.EI
        EA = self.EA

        x = np.linspace(0.0, L, num_points)

        local_disp=np.matmul(self.T, u_global)
        w_1 = local_disp[1]
        phi_1= local_disp[2]
        w_2 = local_disp[4]
        phi_2 = 0

        u = qx*(-L*x/(2*EA) + x**2/(2*EA))+local_disp[3]*(x/L)+local_disp[0]*(1-x/L)

        # w = phi1*(-x + 2*x**2/L - x**3/L**2) + phi2*(x**2/L - x**3/L**2) + q*(L**2*x**2/(24*EI) - L*x**3/(12*EI) + x**4/(24*EI)) + w1*(1 - 3*x**2/L**2 + 2*x**3/L**3) + w2*(3*x**2/L**2 - 2*x**3/L**3)
        w =(EI*L**3*(-phi_1*x + w_1) + L**3*q*x**4/24 + L*x**2*(24*EI*L*phi_1 - 24*EI*w_1 + 24*EI*w_2 + L**4*q)/16 + x**3*(-24*EI*L*phi_1 + 24*EI*w_1 - 24*EI*w_2 - 5*L**4*q)/48)/(EI*L**3)

        return u, w
    
    def plot_moment_diagram (self, u_elem, num_points=10, global_c=False, scale=1.0):
        """
        Plots the bending moment diagram of the element.

        Args:
            u_global (numpy.ndarray): Global displacement vector of the element.
            num_points (int, optional): Number of points to calculate the bending moments. Default is 2.
            global_c (bool, optional): If True, plots the bending moment diagram in the global coordinate system. Default is False (plots in local coordinate system).
            scale (float, optional): Scale factor for the bending moment diagram. Default is 1.0.

        Returns:
            None
        """
        import matplotlib.pyplot as plt

        x = np.linspace ( 0.0, self.L, num_points )
        M = self.bending_moments ( u_elem, num_points )
        xM_local = np.vstack((np.hstack([0,x,x[-1]]),np.hstack([0,M,0])*scale))
        if global_c:
            xM_global = np.matmul(self.Tt[0:2,:2],xM_local)
            xz_start_node = np.vstack((np.ones(num_points+2)*self.nodes[0].x, np.ones(num_points+2)*self.nodes[0].z))
            xz_Mlijn = xM_global + xz_start_node
            p = plt.plot(xz_Mlijn[0,:],xz_Mlijn[1,:])
            X0= self.nodes[0].x
            Z0= self.nodes[0].z
            X1= self.nodes[1].x
            Z1= self.nodes[1].z
            plt.plot((X0, X1), (Z0, Z1), color=p[0].get_color())
            plt.axis('off')
            plt.axis('equal')
        else:
            p = plt.plot(xM_local[0,:],xM_local[1,:])
            plt.xlabel ( "x" )
            plt.ylabel ( "M" )
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.title('Moment line')

    def plot_displaced(self, u_elem, num_points=10, global_c=False, scale=1.0):
        """
        Plots the displacd element.

        Args:
            u_global (numpy.ndarray): Global displacement vector of the element.
            num_points (int, optional): Number of points to calculate the bending moments. Default is 2.
            global_c (bool, optional): If True, plots the displacement diagram in the global coordinate system. Default is False (plots in local coordinate system).
            scale (float, optional): Scale factor for the displacement diagram. Default is 1.0.

        Returns:
            None
        """

        x = np.linspace ( 0.0, self.L, num_points )
        u, w = self.full_displacement ( u_elem, num_points )
        uw_local = np.vstack((x+u*scale,w*scale))
        if global_c:
            uw_global = np.matmul(self.Tt[:2,:2],uw_local)
            xz_start_node = np.vstack((np.ones(num_points)*self.nodes[0].x, np.ones(num_points)*self.nodes[0].z))
            uw = uw_global + xz_start_node
            p =  plt.plot(uw[0,:],uw[1,:])
            X0= self.nodes[0].x
            Z0= self.nodes[0].z
            X1= self.nodes[1].x
            Z1= self.nodes[1].z
            plt.plot((X0, X1), (Z0, Z1), color=p[0].get_color(),alpha=0.3)
            plt.axis('off')
            plt.axis('equal')
        else:
            p = plt.plot(uw_local[0,:],uw_local[1,:])
            plt.plot((0, self.L), (0, 0), color=p[0].get_color(),alpha=0.3)
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.title('Displaced structure')

    def __str__(self):
        """
        Returns a string representation of the Element object.
        
        The string includes the values of the node1, node2 attributes.
        """
        return f"Element connecting:\nnode #1:\n {self.nodes[0]}\nwith node #2:\n {self.nodes[1]}"