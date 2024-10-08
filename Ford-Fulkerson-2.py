from collections import deque

class FlowEdge:
    def __init__(self, u, v, capacity=1, flow=0) -> None:
        self.u = u  
        self.v = v  
        self.capacity = capacity  
        self.flow = 0  

    def update_flow(self, added_flow):
        """
        Update the flow in the edge.
        """
        self.flow += added_flow
        if self.flow > self.capacity:
            raise ValueError("Flow exceeds capacity")

    def __str__(self):
        return f"{self.u.id} -> {self.v.id} (capacity: {self.capacity}, flow: {self.flow})"

class ResidualEdge:
    def __init__(self, u, v, w=0, is_forward=True) -> None:
        self.u = u  
        self.v = v  
        self.is_forward = is_forward

        if self.is_forward: # if this is a forward edge
            self.w = w  # remaining capacity
            self.reverse_edge = None  # this edge has a backward edge
            self.flow_edge = None # the edge in the flow network linked to forward edge in the residual network
        else:
            self.w = w  # undo flow
            self.reverse_edge = None  # this edge has a forward edge

    def update_flow(self, added_flow):
        """
        Update the flow in the edge.
        """
        if self.is_forward:
            self.w -= added_flow
            self.reverse_edge.w += added_flow # update the undo flow
            self.flow_edge.update_flow(added_flow) # update the flow in the flow network
        else: # if this is a backward edge
            self.w += added_flow # update the undo flow
            self.reverse_edge.flow -= added_flow # update the flow in the forward edge
            self.reverse_edge.flow_edge.update_flow(added_flow) # update the flow in the flow network

        if self.w < 0:
            raise ValueError("Flow exceeds residual capacity")

    def __str__(self):
        if self.is_forward:
            return f"{self.u.id} -> {self.v.id} (balance: {self.w})"
        else:
            return f"{self.u.id} -> {self.v.id} (undo: {self.w})"
    
class Vertex:
    def __init__(self, id) -> None:
        self.id = id
        self.edges = []
        self.previous = None       
        self.visited = False
        self.discovered = False

    def add_edge(self, edge):
        self.edges.append(edge)

    def __str__(self):
        return self.id
    
class FlowNetwork:
    def __init__(self, N: int, source: int, sink: int) -> None:
        self.source = source
        self.sink = sink

        self.vertices = [None] * N
        for i in range(N):
            self.vertices[i] = Vertex(i)

    def add_edge(self, u_id: int, v_id: int, capacity: int, flow: int) -> None:
        u = self.vertices[u_id]
        v = self.vertices[v_id]
        
        edge = FlowEdge(u, v, capacity, flow)  
        u.add_edge(edge)

    def reset_visits(self):
        for vertex in self.vertices:
            vertex.visited = False
            vertex.discovered = False
            vertex.previous = None

class ResidualNetwork:
    def __init__(self, flow_network: FlowNetwork) -> None:
        self.flow_network = flow_network
        self.source_id = flow_network.source 
        self.sink_id = flow_network.sink

        # Create a list of vertices
        self.vertices = [Vertex(v.id) for v in flow_network.vertices]  # O(V)
        self.create_residual_network()

    def create_residual_network(self):
        """
        Create the residual network based on the flow network
        """
        for vertex in self.flow_network.vertices:
            for edge in vertex.edges:
                # Add the forward edge and the backward edge
                self.add_edges(edge)

    def add_edges(self, edge: FlowEdge) -> None:
        """
        Add the forward edge and the backward edge to the residual network.
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

        u.add_edge(forward_edge)
        v.add_edge(backward_edge)
        
    def bfs(self):
        """
        Use BFS to find an augmenting path in the residual network.
        """
        self.reset_visits()

        start_vertex = self.vertices[self.source_id]
        discovered = deque([start_vertex])

        while discovered:
            current_vertex = discovered.popleft()

            # if we reach the sink, we have found the path
            if current_vertex.id == self.sink_id:
                path = []
                while current_vertex.id != self.source_id:
                    edge = current_vertex.previous
                    path.append(edge)
                    current_vertex = edge.u if edge.v == current_vertex else edge.v
                path.reverse()
                return path

            current_vertex.visited = True

            # traverse the neighbors
            for edge in current_vertex.edges:
                neighbor = edge.v if edge.u == current_vertex else edge.u

                if edge.w > 0 and not neighbor.visited:
                    neighbor.previous = edge
                    discovered.append(neighbor)

        return None  # return None if no path is found

    def has_aumenting_path(self):
        """
        Check if there is an augmenting path in the residual network.
        """
        path = self.bfs()
        if path is None:
            return False
        else:
            return True
    
    def get_augmenting_path(self):
        """
        Get the augmenting path and the minimum flow in the path.
        """
        path = self.bfs()

        if not path:
            return None, 0

        # find the minimum flow in the path
        min_flow = float('inf')
        for edge in path:
            min_flow = min(min_flow, edge.w)

        return path, min_flow
    
    def update_residual_network(self, path, min_flow):
        """
        Update the residual network based on the augmenting path and the minimum flow.
        """
        for edge in path:
            edge.update_flow(min_flow)  # update the flow
    
    def reset_visits(self):
        for vertex in self.vertices:
            vertex.visited = False
            vertex.discovered = False  
            vertex.previous = None

    def ford_fulkerson(self):
        """
        Find the maximum flow in the flow network using the Ford-Fulkerson algorithm.
        """
        flow = 0

        # as long as there is an augmenting path, we can continue
        while self.has_aumenting_path():
            # take the path and the minimum flow
            path, min_flow = self.get_augmenting_path()

            # augment the flow equal to the residual capacity 
            flow += min_flow

            # update the residual network
            self.update_residual_network(path, min_flow)
          
        return flow

if __name__ == "__main__":
    # Test case: Construct the flow network based on the provided test case.

    N = 7
    source = 1  # s
    sink = 6   # t

    # Create the flow network and add edges based on the provided capacities
    flow_network = FlowNetwork(N, source, sink)

    for vertex in flow_network.vertices:
        for edge in vertex.edges:
            print(edge)

    print("----------------------------------------------------------")


    # Add edges
    flow_network.add_edge(1, 2, capacity=16, flow=0)  # s -> a
    flow_network.add_edge(1, 3, capacity=13, flow=0)  # s -> b
    flow_network.add_edge(2, 3, capacity=10, flow=0)  # a -> b
    flow_network.add_edge(2, 4, capacity=12, flow=0)  # a -> c
    flow_network.add_edge(3, 2, capacity=4, flow=0)   # b -> a
    flow_network.add_edge(3, 5, capacity=14, flow=0)  # b -> d
    flow_network.add_edge(4, 3, capacity=9, flow=0)   # c -> b
    flow_network.add_edge(4, 6, capacity=20, flow=0)  # c -> t
    flow_network.add_edge(5, 4, capacity=7, flow=0)   # d -> c
    flow_network.add_edge(5, 6, capacity=4, flow=0)   # d -> t

    # Create the residual network based on the flow network
    residual_network = ResidualNetwork(flow_network)

    flow = residual_network.ford_fulkerson()
    print(flow)  

    for vertex in flow_network.vertices:
        for edge in vertex.edges:
            print(edge)

    print("----------------------------------------------------------")

    for vertex in residual_network.vertices:
        for edge in vertex.edges:
            print(edge)