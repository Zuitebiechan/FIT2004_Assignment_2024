# ========================================================
# FIT2004 Assignment 2 2024 Semester 1
# Name: Haoshuang Song
# Student ID: 34517812
# ========================================================

from collections import deque

# ======================================================== Q1 Code =================================================================
class Vertex:
    """ 
    ----------------------------------------------- Class for Vertex ----------------------------------------------------
    The Vertex object is used to represent the data centres. 

    Attributes:
        id:                 A unique id to distinguish vertices apart
        edges:              A list of outgoing Edges of Vertex
        discovered:         A boolean value which states if the Vertex has been discovered
        visited:            A boolean value which states if the Vertex has been visited
        previous:           The previous Edge (use for backtracking the path from source)

    """
    def __init__(self, id: int) -> None:
        """
        Description: 
            Constructor for Vertex object.
        Input:
            id: An integer value, representing a unique id to distinguish vertices apart
        Time complexity: 
            Best and Worst: O(1)
        Aux space complexity: 
            In-place, O(1)
        """
        self.id = id
        self.edges = []
        self.previous = None       
        self.visited = False
        self.discovered = False

    def add_edge(self, edge):
        """
        Description: 
            Append Edge object into list of edges.
        Input:
            edge: Edge object (outgoing edge from the Vertex)
        Time complexity: 
            Best and Worst: O(1)
        Aux space complexity: 
            In-place, O(1)
        """
        self.edges.append(edge)

    def __str__(self):
        """
        Description:
            Return the id of the Vertex object.
        Output:
            A string representation of the Vertex object.
        Time complexity:
            Best and Worst: O(1)
        Aux space complexity:
            In-place, O(1)
        """
        return str(self.id)

class FlowEdge:
    """ 
    ----------------------------------------------- Class for FlowEdge ----------------------------------------------------
    The Edge object is used to represent the outgoing edges with maximum amount of flow and current flow 
    in the flow network.

    Attributes:
        u:                  Integer value representing the id of starting Vertex of the Edge 
        v:                  Integer value representing the id of ending Vertex of the Edge
        capacity:           Integer value representing the maximum amount of flow on the Edge
        flow:               Integer value representing the current amount of flow on the Edge    
    """
    def __init__(self, u, v, capacity=1, flow=0) -> None:
        """
        Description: 
            Constructor for FlowEdge object.
        Input:
            u:              Integer value representing the id of starting Vertex of the Edge 
            v:              Integer value representing the id of ending Vertex of the Edge
            capacity:       Integer value representing the maximum amount of flow on the Edge (default 1)
            flow:           Integer value representing the current amount of flow on the Edge (default 0)
                            i.e. u -> v (5/10) where 5 is the flow and 10 is the capacity
        Time complexity: 
            Best and Worst: O(1)
        Aux space complexity: 
            In-place, O(1)
        """
        self.u = u  
        self.v = v  
        self.capacity = capacity  
        self.flow = 0  

    def update_flow(self, added_flow) -> None:
        """
        Description:
            Update the current flow in the edge.
        Input:
            added_flow:     Integer value representing the amount of flow to be added
        Exceptions:
            ValueError:     Raised when the flow exceeds the capacity or the flow is negative
        Time complexity:
            Best and Worst: O(1)
        Aux space complexity:
            In-place, O(1)
        """
        self.flow += added_flow

        if self.flow > self.capacity:
            raise ValueError("Flow exceeds capacity")
        if self.flow < 0:
            raise ValueError("Flow cannot be negative")

    def __str__(self) -> str:
        """
        Description:
            Return the string representation of the FlowEdge object.
        Output:
            A string representation of the FlowEdge object.
            i.e. 0 -> 1 (capacity: 1, flow: 0)
        Time complexity:
            Best and Worst: O(1)
        Aux space complexity:  
            In-place, O(1)
        """
        return f"{self.u.id} -> {self.v.id} (capacity: {self.capacity}, flow: {self.flow})"

class ResidualEdge:
    """ 
    ----------------------------------------------- Class for ResidualEdge ----------------------------------------------------
    The Edge object is used to represent the forward and backward edges in the residual network linking to the flow edges 
    in the flow network.

    Attributes:
        u:                  Integer value representing the id of starting Vertex of the Edge 
        v:                  Integer value representing the id of ending Vertex of the Edge
        w:                  Integer value representing the remaining capacity or undo flow on the forward or backward Edge
        is_forward:         Boolean value representing if the Edge is a forward edge or a backward edge    
    """
    def __init__(self, u, v, w=0, is_forward=True) -> None:
        """
        Description: 
            Constructor for ResidualEdge object.
        Input:
            u:              Integer value representing the id of starting Vertex of the Edge 
            v:              Integer value representing the id of ending Vertex of the Edge
            w:              Integer value representing the remaining capacity or undo flow on the forward or backward Edge (default 0)
            is_forward:     Boolean value representing if the Edge is a forward edge or a backward edge (default True)
        Time complexity:
            Best and Worst: O(1)
        Aux space complexity:
            In-place, O(1)
        """
        self.u = u  
        self.v = v  
        self.is_forward = is_forward

        if self.is_forward:
            # if this is a forward edge 
            self.w = w  # remaining capacity (= capacity - flow)
            self.reverse_edge = None  # linking to the backward edge
            self.flow_edge = None # the forward edge in the residual network linked to the edge in the flow network
        else:
            # if this is a backward edge
            self.w = w  # undo flow (= flow)
            self.reverse_edge = None  # linking to the forward edge

    def update_flow(self, new_flow):
        """
        Description:
            Update the balance or undo flow on the forward or backward Edge.
        Input:
            new_flow:       Integer value representing the amount of flow to be deducted
        Exceptions:
            ValueError:     Raised when the new flow exceeds the balance or undo flow
        Time complexity:
            Best and Worst: O(1)
        Aux space complexity:
            In-place, O(1)
        """
        if self.w < new_flow: # balance or undo flow should be greater than the new flow
            raise ValueError("Flow exceeds residual capacity")
        
        # if this is a forward edge, balance = balance - new flow, undo = undo + new flow
        # if this is a backward edge, undo = undo - new flow, balance = balance + new flow
        self.w -= new_flow
        self.reverse_edge.w += new_flow
        
        # update the flow of the edge in the flow network 
        # O(1)
        if self.is_forward:
            # if the flow goes through a forward edge, means the flow should be added in the flow network
            self.flow_edge.update_flow(new_flow) 
        else:
            # if the flow goes through a backward edge, means the flow should be deducted in the flow network
            self.reverse_edge.flow_edge.update_flow(-new_flow) 

    def __str__(self):
        """
        Description:
            Return the string representation of the ResidualEdge object.
        Output:
            A string representation of the ResidualEdge object.
            i.e. 0 -> 1 (balance: 1); 1 -> 0 (undo: 0)
        Time complexity:
            Best and Worst: O(1)
        Aux space complexity:
            In-place, O(1)
        """
        if self.is_forward:
            return f"{self.u.id} -> {self.v.id} (balance: {self.w})"
        else:
            return f"{self.u.id} -> {self.v.id} (undo: {self.w})"
    
class FlowNetwork:
    """ 
    ----------------------------------------------- Class for Flow Network ------------------------------------------------------------
    The FlowNetwork object represent relationships between participants and activities in a flow network.

    Vertices represent participants and activities along with a source and sink vertex, 
        The first two vertices represent the source and sink.
        The next N vertices represent the participants.
        The next M vertices represent the activities that must have two leaders.
        The next M vertices represent the activities that do not must have a leader.
        
    and Edges represent the preferences of participants to activities.
        From source to participants, each participant has an edge with capacity 1.
        From participants to activities that must have two leaders, each participant has an edge with capacity 1 to those activities if they have experience on it.
            (preferences[i][j] == 2)
        From participants to activities that do not must have a leader, each participant has an edge with capacity 1 to those activities if they are interested in it.
            (preferences[i][j] == 1)
        From activities that must have two leaders to sink, each activity has an edge with capacity 2, meaning it must have two leaders.
        From activities that do not must have a leader to sink, each activity has an edge with capacity places[i]-2, meaning each activity[i] can have at most places[i] participants.

    Flow Network will be like (wihout edges between participants and activities):
          |----1---p_1(2)       la_1(N+1)----------2-----------|
          |----1---p_2(3)       la_2(N+2)----------2-----------|
          |        .            .                              |
          |        .            .                              |
          |        .            .                              |
          |        .            la_M(N+M)----------2-----------|
        s(0)       .                                          t(1)
          |        .            ia_1(N+M+1)--(places[0]-2)-----|
          |        .            ia_2(N+M+2)--(places[1]-2)-----|
          |        .            .                              |
          |        .            .                              |
          |----1---p_N(N)       ia_M(N+2M)---(places[M]-2)-----|

          where
            s is the source vertex, 
            t is the sink vertex,
            p_i is the i-th participant, 
            la_i is the i-th activity that must have two leaders, 
            ia_i is the i-th activity that do not must have a leader,
            the number behind the vertex is the id of the vertex and also the index in the vertices list,
            the number on the edge is the capacity of the edge.

    Attributes:
        source:                     Interger 0, represent the id of source vertex.
        sink:                       Interger 1, represent the id of sink vertex.
        max_participant:            An integer value, represent number of participants.
        max_activity:               An integer value, represent number of activities.
        vertices_count:             An integer value, represent number of vertices needed for flow network.
                                    It is the 2 + self.max_participant + 2 * self.max_activity.
        vertices:                  A list of Vertex object of length vertices_count.
                                            
                                    Index                       Item
                                    0                           source
                                    1                           sink
                                    2...N                       participants
                                    N+1...N+M                   leader_activity
                                    N+M+1...N+2M                interested_activity

                                    [source, sink, participant_1, participant_2, ... , participant_N, leader_activity_1, leader_activity_2, ... , leader_activity_M, interested_activity_1, interested_activity_2, ... , interested_activity_M]
    """

    def __init__(self, preferences: list[list], places: list[list]) -> None: # O(N^2)
        """
        Description:
            Constructor for FlowNetwork object.
        Input:
            preferences:        A list of lists, where each sublist contains the preferences of a participant.
                                The preferences are represented as integers:
                                    0: Not interested in the activity
                                    1: Interested in the activity
                                    2: Experienced in the activity
            places:             A list of integers, where each integer represents the number of places available for an activity.
        Time complexity:
            Best and Worst: O(N*M) -> O(N^2) where N is the number of participants (M is at most N/2)
        Analysis:
            The add_edges method has a time complexity of O(N^2) because it iterates over all participants and activities.
        Aux space complexity:
            O(2 + N + 2M) -> O(N + 3M) where N is the number of participants and M is the number of activities.
        Analysis:
            The vertices list has a length of 2 + N + 2M.
        """
        self.source = 0
        self.sink = 1
        self.max_participant = len(preferences)
        self.max_activity = len(places)
        self.vertices_count = 2 + self.max_participant + 2 * self.max_activity
        self.vertices = [None] * (self.vertices_count)

        # Define the index of the first participant, leader node, and interested node
        self.index_to_first_participant = 2
        self.index_to_first_leader_node = self.index_to_first_participant + self.max_participant
        self.index_to_first_interested_node = self.index_to_first_leader_node + self.max_activity

        for i in range(self.vertices_count): # O(2 + N + 2M) -> O(N + 3M)
            self.vertices[i] = Vertex(i) 

        self.add_edges(preferences, places) # O(N^2)

    def add_edges(self, preferences, places) -> None:
        """
        Description:
            Add edges to the flow network based on the preferences and places available.
        Input:
            preferences:        A list of lists, where each sublist contains the preferences of a participant.
            places:             A list of integers, where each integer represents the number of participants assigned to an activity.
        Time complexity:
            Best and Worst: O(N*M) -> O(N^2) where N is the number of participants (M is at most N/2)
        Analysis:
            The method iterates over all participants and activities to add edges
        Aux space complexity:
            In-place, O(1)
        """
        # Add edges from source to participants
        source = self.vertices[0]
        for i in range(self.max_participant): # O(N)
            participant = self.vertices[i + 2] # the first participant node starts from 2 cuz 0 and 1 are source and sink
            edge = FlowEdge(u=source, v=participant, capacity=1) # source -> participant
            source.add_edge(edge) # O(1)

        # Add edges from participant[i] to leader_activity[i] and interested_activity[i] # O(N * M) -> O(N^2)
        for i in range(self.max_participant): # O(N)
            for j in range(self.max_activity): # O(M)
                participant = self.vertices[i + self.index_to_first_participant] 
                if preferences[i][j] == 0:
                    # if participant is not interested in any activity, no need to add edges
                    continue
                elif preferences[i][j] == 1:
                    # if participant is interested in the activity, add edge to interested_activity
                    activity_interested = self.vertices[j + self.index_to_first_interested_node] # the first activity node starts from max_participant + 2
                    edge = FlowEdge(u=participant, v=activity_interested, capacity=1) # participant -> activity_interested
                    participant.add_edge(edge)
                elif preferences[i][j] == 2:
                    # if participant is experienced in the activity, add edge to leader_activity
                    activity_leader = self.vertices[j + self.index_to_first_leader_node]
                    edge = FlowEdge(u=participant, v=activity_leader, capacity=1) # participant -> activity_experienced
                    participant.add_edge(edge)
        
        # Add edges from leader_activity[i] node and interested_activity[i] to sink node 
        for i in range(self.max_activity): # O(M)
            activity_leader = self.vertices[i + self.index_to_first_leader_node]
            activity_interested = self.vertices[i + self.index_to_first_interested_node]
            sink = self.vertices[1]
            edge = FlowEdge(u=activity_leader, v=sink, capacity=2) # activity_experienced -> activity
            activity_leader.add_edge(edge)
            edge = FlowEdge(u=activity_interested, v=sink, capacity=places[i]-2) # activity_interested -> activity
            activity_interested.add_edge(edge)

    def get_assignments(self) -> list[list]:
        """
        Description:
            Get the assignments of participants to activities based on the flow network.
        Output:
            A list of lists, where each sublist contains the participants assigned to an activity.
        Time complexity:
            Best and Worst: O(N*M) -> O(N^2) where N is the number of participants (M is at most N/2)
        Analysis:
            The method iterates over all participants and activities to get the assignments
        Aux space complexity:
            O(N) where N is the number of participants
        Analysis:
            The method uses a list of lists with a length of N to store the assignments
        """
        assignments = [[] for _ in range(self.max_activity)]  # initialize the assignments list

        # Iterate over all participants
        for i in range(self.max_participant): # O(N)
            participant_vertex = self.vertices[self.index_to_first_participant + i]
            participant_id = i  # the id of the participant

            # Iterate over all edges of the participant
            for edge in participant_vertex.edges: # O(M)
                if edge.flow > 0:
                    # edge.v is the activity vertex (leader or interested)
                    activity_vertex = edge.v
                    activity_index = None
                    
                    # Check if the activity vertex is a leader or interested node
                    if self.index_to_first_leader_node <= activity_vertex.id < self.index_to_first_leader_node + self.max_activity:
                        # leader node
                        activity_index = activity_vertex.id - self.index_to_first_leader_node
                        
                    elif self.index_to_first_interested_node <= activity_vertex.id < self.index_to_first_interested_node + self.max_activity:
                        # interested node
                        activity_index = activity_vertex.id - self.index_to_first_interested_node

                    if activity_index is not None:
                        # add the participant to the activity
                        assignments[activity_index].append(participant_id) # O(1)

        return assignments

class ResidualNetwork:
    """ 
    ----------------------------------------------- Class for Residual Network ------------------------------------------------------------
    The ResidualNetwork object represent the residual network of the flow network.

    Attributes:
        flow_network:               FlowNetwork object, represent the flow network.
        source_id:                  Integer value, represent the id of source vertex.
        sink_id:                    Integer value, represent the id of sink vertex.
        vertices:                   A list of Vertex object of length of the number of vertices in the flow network.
    """
    def __init__(self, flow_network: FlowNetwork) -> None:
        """
        Description:
            Constructor for ResidualNetwork object.
        Input:
            flow_network:           FlowNetwork object, represent the flow network.
        Time complexity:
            Best and Worst: O(V+E) where V is the number of vertices and E is the number of edges in the flow network.
        Analysis:
            The method calls the create_residual_network method which has a time complexity of O(V+E).
        Aux space complexity:
            O(V) where V is the number of vertices in the flow network.
        Analysis:
            The method uses a list of vertices with a length of V to store the vertices
        """
        self.flow_network = flow_network
        self.source_id = flow_network.source 
        self.sink_id = flow_network.sink

        # Create a list of vertices
        self.vertices = [Vertex(v.id) for v in flow_network.vertices]  # O(V)

        self.create_residual_network() # O(V + E)

    def create_residual_network(self) -> None:
        """
        Description:
            Create the residual network based on the flow network.
        Time complexity:
            Best and Worst: O(V + E) where V is the number of vertices and E is the number of edges in the flow network.
        Analysis:
            The method iterates over all vertices and edges in the flow network to create the residual network.
        Aux space complexity:
            O(E) where E is the number of edges in the flow network.
        """
        for vertex in self.flow_network.vertices:
            for edge in vertex.edges:
                # Add the forward edge and the backward edge
                self.add_edges(edge) # O(1)

    def add_edges(self, edge: FlowEdge) -> None:
        """
        Description:
            Add the forward edge and the backward edge to the residual network.
        Input:
            edge: FlowEdge object, represent the edge in the flow network.
        Time complexity:
            Best and Worst: O(1)
        Analysis:
            The method creates two ResidualEdge objects and adds them to the vertices list which takes O(1) time.
        Aux space complexity:
            In-place, O(1)
        """
        u = self.vertices[edge.u.id]
        v = self.vertices[edge.v.id]

        forward_edge = ResidualEdge(u, v, w=(edge.capacity - edge.flow), is_forward=True)
        backward_edge = ResidualEdge(v, u, w=edge.flow, is_forward=False)

        # Set the reverse edge linked to the forward edge
        forward_edge.reverse_edge = backward_edge
        forward_edge.flow_edge = edge

        # Set the reverse edge linked to the backward edge
        backward_edge.reverse_edge = forward_edge

        u.add_edge(forward_edge) # O(1)
        v.add_edge(backward_edge)
        
    def bfs(self) -> list:
        """
        Description:
            Perform a breadth-first search to find an augmenting path in the residual network.
        Output:
            A list of edges representing the augmenting path.
        Time complexity:
            Best: O(V) where V is the number of vertices in the residual network.
            Worst: O(V + E) where V is the number of vertices and E is the number of edges in the residual network.
        Analysis:
            Best: The method returns the augmenting path in the first iteration.
            Worst: The method traverses the vertices and edges in the residual network to find the augmenting path.
        Aux space complexity:
            O(V) where V is the number of vertices in the residual network.
        Analysis:
            The method uses a deque to store the discovered vertices.
        """
        self.reset_visits()

        start_vertex = self.vertices[self.source_id]
        discovered = deque([start_vertex]) 

        while discovered: # O(V)
            current_vertex = discovered.popleft() # O(1)

            # if we reach the sink, we have found the path
            if current_vertex.id == self.sink_id:
                path = []
                while current_vertex.id != self.source_id:
                    edge = current_vertex.previous
                    path.append(edge) # O(1)
                    current_vertex = edge.u 
                path.reverse() # O(V)
                return path

            current_vertex.visited = True

            # traverse the neighbors
            for edge in current_vertex.edges: # O(E)
                if edge.u == current_vertex and edge.w > 0:
                    neighbor = edge.v
                    if not neighbor.visited:
                        neighbor.previous = edge
                        discovered.append(neighbor)

        return None  # return None if no path is found

    def has_augmenting_path(self) -> bool:
        """
        Description:
            Check if there is an augmenting path in the residual network.
        Output:
            A boolean value representing if there is an augmenting path.
        Time complexity:
            Best: O(V) where V is the number of vertices in the residual network.
            Worst: O(V + E) where V is the number of vertices and E is the number of edges in the residual network.
        Analysis:
            Best: The bfs() method returns the augmenting path in the first iteration.
            Worst: The bfs() method traverses the vertices and edges in the residual network to find the augmenting path.
        Aux space complexity:
            O(V) where V is the number of vertices in the residual network.
        Analysis:
            The bfs() method uses a deque to store the discovered vertices.
        """
        path = self.bfs() # Best: O(V), Worst: O(V + E)
        if path is None:
            return False
        else:
            return True
    
    def get_augmenting_path(self) -> tuple:
        """
        Description:
            Get the augmenting path and the minimum flow in the path.
        Output:
            A tuple containing the augmenting path and the minimum flow in the path.
        Time complexity:
            Best: O(V) where V is the number of vertices in the residual network.
            Worst: O(V + E) where V is the number of vertices and E is the number of edges in the residual network.
        Analysis:
            Best: The bfs() method returns the augmenting path in the first iteration.
            Worst: The bfs() method traverses the vertices and edges in the residual network to find the augmenting path.
        Aux space complexity:
            O(V) where V is the number of vertices in the residual network.
        Analysis:
            The bfs() method uses a deque to store the discovered vertices.
        """
        path = self.bfs()

        if not path:
            return None, 0

        # find the minimum flow in the path
        min_flow = float('inf')
        for edge in path:
            min_flow = min(min_flow, edge.w)

        return path, min_flow
    
    def update_residual_network(self, path: list[ResidualEdge], min_flow: int) -> None:
        """
        Description:
            Update the residual network based on the augmenting path and the minimum flow.
        Input:
            path:               A list of edges representing the augmenting path.
            min_flow:           An integer value representing the minimum flow in the path.
        Time complexity:
            Best and Worst: O(V) where V is the number of vertices in the residual network.
        Analysis:
            The method iterates over all edges in the path to update the flow.
        Aux space complexity:
            In-place, O(1)
        """
        for edge in path:
            edge.update_flow(min_flow)  # O(1)
    
    def reset_visits(self) -> None:
        """
        Description:
            Reset the visited, discovered and previous attributes of all vertices in the residual network.
        Time complexity:
            Best and Worst: O(V) where V is the number of vertices in the residual network.
        Analysis:
            The method iterates over all vertices in the residual network to reset the attributes.
        Aux space complexity:
            In-place, O(1)
        """
        for vertex in self.vertices:
            vertex.visited = False
            vertex.discovered = False  
            vertex.previous = None

    def ford_fulkerson(self) -> int:
        """
        Description:
            Implement the Ford-Fulkerson algorithm to find the maximum flow in the residual network.
        Output:
            An integer value representing the maximum flow in the flow network.
        Time complexity:
            Best: O(V+E) where V is the number of vertices in the residual network and E is the number of edges in the residual network.
            Worst: O(FV + FE) where V is the number of vertices in the residual network, E is the number of edges in the residual network, and F is the maximum flow.
        Analysis:
            Best: The method finds the maximum flow in the first iteration.
            Worst: The method iterates over all augmenting paths to find the maximum flow while every iteration the maximum flow increases by 1.
        Aux space complexity:
            O(V) where V is the number of vertices in the residual network.
        Analysis:
            The method calls the has_augmenting_path() and get_augmenting_path() which have an aux space complexity of O(V).
        """
        flow = 0

        # as long as there is an augmenting path, we can continue
        while self.has_augmenting_path():
            # take the path and the minimum flow
            path, min_flow = self.get_augmenting_path()

            # augment the flow equal to the residual capacity 
            flow += min_flow

            # update the residual network
            self.update_residual_network(path, min_flow)
          
        return flow

def assign(preferences: list[list], places: list[list]) -> list[list]:
    """
    Description:
        Assign participants to activities based on their preferences and the number of places available.
    Input:
        preferences:        A list of lists, where each sublist contains the preferences of a participant.
        places:             A list of integers, where each integer represents the number of participants assigned to an activity.
    Output:
        A list of lists, where each sublist contains the participants assigned to an activity.
    Time complexity:
        Best: O(N^2) where N is the number of participants.
        Worst: O(N^3) where N is the number of participants.
    Analysis:
        Best: The method calls ford_fulkerson() with its best time complexity of O(V+E) which is O(N^2) in this case.
        Worst: The method calls ford_fulkerson() with its worst time complexity of O(FV + FE) which is O(N^3) in this case.
            In this case: 
                V is the number of vertices in the flow network which is 2 + N + 2M             
                E is double the number of edges in the flow network which is 2(N + NM + 2M) (in worst senario, each participant has no disinterested acticities)
                where N is the number of participants and M is the number of activities;   
    Aux space complexity:
        O(2 + N + 2M) where N is the number of participants and M is the number of activities.
    Analysis:
        The method calls ford_fulkerson() which has an aux space complexity of O(V) where V = 2 + N + 2M in this case.
    """
    flow_network = FlowNetwork(preferences, places);
    residual_network = ResidualNetwork(flow_network);

    # Best: O(V+E), Worst: O(FV + FE) -> Best: O(N^2), Worst: O(N^3)
    # In this case, O(V) = O(2 + N + 2M) = O(N), O(E) = O(2(N + NM +2M)) = O(N^2), O(F) = O(N) 
    max_flow = residual_network.ford_fulkerson(); 

    # check if the maximum flow is equal to the no. of participants
    if max_flow != sum(places):  # O(M)，where M is the number of activities
        return None  

    # get the assignments
    solution = flow_network.get_assignments() # O(N*M) -> O(N^2)

    return solution

# ======================================================== Q2 Code =================================================================
