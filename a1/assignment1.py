"""
Assignment 1, FIT2004
================================
Name: Haoshuang Song
Student ID: 34517812
================================
Due: 06/09/2024
"""

class CityMap:
    """
    Class description:
    The CityMap class represents a transportation network with both roads (bidirectional) and tracks (unidirectional). 
    The class manages vertices (locations) and allows the user to calculate optimal routes with possible friend pickups.
    
    Methods:
        - __init__(self, roads, tracks, friends): Initializes the city map with roads, tracks, and friends located at certain vertices.
        - plan(self, start, destination): Calculates the best route from a starting point to a destination, optionally picking up a friend along the way.
        - calculate_friend_pickups(self, friends, tracks, max_depth): Finds possible friend pickup locations based on a BFS within the given depth.
    """
    def __init__(self, roads: list[tuple[int, int, int]], tracks: list[tuple[int, int, int]], friends: list[tuple[str, int]]) -> None:  
        """
        Function description:
        Initializes the CityMap with vertices connected by roads and tracks, and places friends at specified locations. It also constructs the graph representation for the city.
        
        :Input:
            roads: List of tuples representing bidirectional roads, where each tuple contains (start_vertex_id, end_vertex_id, weight).
            tracks: List of tuples representing unidirectional tracks, where each tuple contains (start_vertex_id, end_vertex_id, weight).
            friends: List of tuples representing friends and their locations, where each tuple contains (friend_name, location_vertex_id).
        
        :Output, return or postcondition:
            Initializes the `self.city_graph` object and populates it with vertices, roads, and tracks.
        
        :Time complexity:
            O(R + T)
        
        :Time complexity analysis:
            Initializing roads takes O(R) and initializing tracks takes O(T) because we iterate over the road and track lists,
            where R is the number of roads and T is the number of tracks.
        
        :Aux space complexity:
            O(L)
        
        :Aux space complexity analysis:
            The space complexity is driven by the graph structure that is based on the number of locations (vertices) `L`.
        """
        # Calculate the maximum vertex ID considering only roads and tracks 
        max_road = max((max(road[0], road[1]) for road in roads), default=0)  # O(R)
        max_track = max((max(track[0], track[1]) for track in tracks), default=0)  # O(T)
        max_vertex = max(max_road + 1, max_track + 1) 

        self.city_graph = Graph(max_vertex + 1) # auxiliary space complexity O(L)
        self.friends = friends  
        self.tracks = tracks
        self.roads = roads
        
        # Add edges from roads 
        for road in roads:  # O(R)
            self.city_graph.add_edge(road[0], road[1], road[2], is_track=False)  # O(1)

        # Add edges from tracks
        for track in tracks:  # O(T)
            self.city_graph.add_edge(track[0], track[1], track[2], is_track=True)  # O(1)


    def plan(self, start: int, destination: int) -> tuple[int, list[int], str, int]:
        """
        Function description:
        Finds the best route from a start location to a destination, optionally picking up a friend on the way. 
        The function returns the best total time, the full route, the picked-up friend (if any), and the pickup location.

        :Input:
            start: The ID of the starting vertex.
            destination: The ID of the destination vertex.

        :Output, return or postcondition:
            Returns a tuple containing:
            - The best total travel time.
            - The path as a list of vertices.
            - The friend picked up along the way (if any).
            - The pickup location.

        :Time complexity:
            O(|R| log |L|)

        :Time complexity analysis:
            Running Dijkstra's algorithm twice (from start and from destination) takes O(|R| log |L|) time, where |L| is the number of locations (vertices) and |R| is the number of roads (edges).
            Checking the possible friend pickups runs in O(F + T), where F is the number of friends and T is the number of tracks. The overall time complexity is dominated by the Dijkstra calls.

        :Aux space complexity:
            O(L)

        :Aux space complexity analysis:
            - pick_list stores the possible friend pickups, which takes O(F + T) space. 
            - dijkstra calls require a min heap with auxiliary space to the number of vertices L.
            - backtracking constructs a path (list) of vertices from the start to the pickup location and from the pickup location to the destination. The maximum size of each path is proportional to the number of vertices, L.
            - The overall space complexity is driven by the graph structure based on the number of locations (vertices) L.
        """ 
        # Step 0: Handle the trivial case where start is the same as the destination
        if start == destination:
            # Check if there's a friend at the start location
            for friend_name, friend_location in self.friends: # O(F)
                if friend_location == start:
                    return 0, [start], friend_name, start  # Immediate return for no movement needed
        
        # Step 1: Calculate possible friend pickups and store in list
        pickup_list = self.calculate_friend_pickups(self.friends, self.tracks, 2)  # O(F + T)

        # Step 2: Run Dijkstra from start to all vertices (updates start_time and start_previous)
        self.city_graph.dijkstra(start, is_start=True)  # O(|R| log |L|)

        # Step 3: Run Dijkstra from destination to all vertices (updates dest_time and dest_previous)
        self.city_graph.dijkstra(destination, is_start=False)  # O(|R| log |L|)

        best_total_time = float('inf')
        best_pickup_location = None
        best_friend_to_pickup = None

        # Step 4: Find the best pickup location based on the total travel time
        for i in range(len(pickup_list)):  # O(L)
            _, pickup_location_id, friend_location = pickup_list[i]  # O(1)
            
            # Get the total time by summing start -> pickup and pickup -> destination times
            start_to_pickup_time = self.city_graph.vertices[pickup_location_id].start_time
            pickup_to_dest_time = self.city_graph.vertices[pickup_location_id].dest_time

            total_time = start_to_pickup_time + pickup_to_dest_time

            # Update if this is the best route so far
            if total_time < best_total_time:
                best_total_time = total_time
                best_pickup_location = pickup_location_id
                best_friend_to_pickup = friend_location.pickup_friend[0]

        # Step 5: If we found a valid pickup point, backtrack to get the path
        if best_pickup_location is not None:
            # Step 6: Get the full route by combining paths from start to pickup and pickup to destination
            route_to_pickup = self.city_graph.backtracking_pickup(best_pickup_location, from_start=True)  # O(L)
            route_from_pickup = self.city_graph.backtracking_pickup(best_pickup_location, from_start=False)  # O(L)

            route_to_pickup.pop()  # Remove the duplicate pickup location # to_pickup: A -> B -> C, to_dest: C -> D -> E, remove C in to_pickup
            route_to_pickup.extend(route_from_pickup)  # Combine paths, A -> B -> C -> D -> E

            return best_total_time, route_to_pickup, best_friend_to_pickup, best_pickup_location

        # Step 7: If no valid pickup location is found, return an invalid result
        return -1, [], None, None

    def calculate_friend_pickups(self, friends: list[tuple[str, int]], tracks: list[tuple[int, int, int]], max_depth: int) -> list:
        """
        Function description:
        Calculates the possible friend pickups using BFS, constrained by a maximum depth. This method returns a list of possible friend pickups.

        :Input:
            friends: A list of tuples representing friends and their locations (friend_name, location_vertex_id).
            tracks: A list of tuples representing unidirectional tracks (start_vertex_id, end_vertex_id, weight).
            max_depth: The maximum number of track moves allowed to find a friend.

        :Output, return or postcondition:
            Returns a list of tuples, where each tuple contains:
            - The number of track moves (depth) to the pickup point.
            - The pickup location ID.
            - The friend at the pickup location.

        :Time complexity:
            O(F + T)

        :Time complexity analysis:
            Processing each friend takes O(F), and the BFS to calculate reachable pickup points through the tracks takes O(T), where F is the number of friends and T is the number of tracks.

        :Aux space complexity:
            O(L)

        :Aux space complexity analysis:
            The space complexity is proportional to the number of locations (L), since pick_list stores the possible friend pickups and this could be up to the number of vertices.
        """
        pickup_list = []
        self.city_graph.reset_visits()  # O(L)

        # add zero-depth friends to the pickup list
        for friend_name, friend_location_id in friends:  # O(F)
            friend_location = self.city_graph.vertices[friend_location_id]
            friend_location.update_friend_pickup_info(friend_name, 0)  # O(1)
            pickup_list.append((0, friend_location.id, friend_location))  # O(1)

        # Calculate reachable pickup points using tracks
        for depth in range(1, max_depth + 1):  # O(max_depth)
            for track in tracks:  # O(T)
                u, v, _ = track
                from_location = self.city_graph.vertices[u]
                to_location = self.city_graph.vertices[v]

                if from_location.pickup_friend is not None and from_location.pickup_friend[1] == depth - 1:
                    if not to_location.visited:
                        to_location.update_friend_pickup_info(from_location.pickup_friend[0], depth)
                        pickup_list.append((depth, to_location.id, to_location))  
                        to_location.visited = True

        return pickup_list  
    
class Graph:
    """
    Class description:
    The Graph class represents a graph structure consisting of vertices and edges. It supports adding directed or undirected edges between vertices and running Dijkstra's algorithm to find the shortest paths. This implementation is optimized to focus on the vertices and edges that are part of the actual connections.

    :Methods:
        __init__(self, N)
        add_edge(self, u_id, v_id, w, is_directed=True, is_track=False)
        dijkstra(self, start_id, is_start=True)
        reset(self)
        reset_visits(self)
        backtracking_pickup(self, target_id, from_start=True)
    """
    def __init__(self, N: int) -> None: 
        """
        Function description:
        Initializes the Graph object with N vertices, pre-allocating space for vertices and initializing each vertex with a unique ID.

        :Input:
            N: The number of vertices in the graph (int).
        :Output, return or postcondition:
            The graph is initialized with N vertices.
        :Time complexity:
            O(N)
        :Time complexity analysis:
            Traverse through N vertices and initialize each vertex with a unique ID.
        :Aux space complexity:
            O(N)
        :Aux space complexity analysis:
            Each vertex takes constant space, so the overall space is proportional to the number of vertices N.
        """
        self.vertices = [None] * N  
        for i in range(N):
            self.vertices[i] = Vertex(i)  

    def add_edge(self, u_id: int, v_id: int, w: int, is_track=False) -> None:
        """
        Function description:
        Adds a directed or undirected edge between two vertices. The edge is stored in the road_edges or track_edges list based on the `is_track` flag. If the edge is undirected, a reverse edge is added.

        :Input:
            u_id: The ID of the starting vertex (int).
            v_id: The ID of the ending vertex (int).
            w: The weight or cost of the edge (int).
            is_track: Boolean flag indicating if the edge is a track edge (default is False).
        :Output, return or postcondition:
            The edge is added to the appropriate list of edges for the two vertices.
        :Time complexity:
            O(1)
        :Time complexity analysis:
            Accessing vertices by their ID is constant time, and appending the edge to the list is also constant.
        :Aux space complexity:
            O(1)
        :Aux space complexity analysis:
            The edge object is created and stored in the vertex's list of edges, requiring constant space.
        """
        u = self.vertices[u_id]
        v = self.vertices[v_id]
        
        edge = Edge(u, v, w)
        u.add_edge(edge, is_track)
        
        # If the edge is a road edge, add a reverse edge
        if not is_track:
            edge_back = Edge(v, u, w)
            v.add_edge(edge_back)

    def dijkstra(self, start_id: int, is_start=True) -> None:
        """
        Function description:
        Implements a modified Dijkstra's algorithm to find the shortest paths from the start vertex to all other connected vertices.
        The algorithm allows revisiting vertices if a shorter path is found after they have been visited. This is achieved using the `update()` method on the MinHeap to adjust the distance of already visited vertices.

        :Input:
            start_id: The ID of the starting vertex (int).
            is_start: Boolean flag indicating if the algorithm is calculating from the start or towards the destination (default is True).
        :Output, return or postcondition:
            Updates the `start_time` and `start_previous` or `dest_time` and `dest_previous` for all reachable vertices.
        :Time complexity:
            O(|R| log |L|)
        :Time complexity analysis:
            Each edge is processed in O(log |L|) due to the priority queue updates, where |R| is the number of edges and |L| is the number of vertices.
            Uses a condition to reduce the number of processed vertices by only processing vertices with outgoing edges.
        :Aux space complexity:
            O(|L|)
        :Aux space complexity analysis:
            Auxiliary space is proportional to the number of vertices, including storage for times, previous pointers, and the heap.
        """
        
        self.reset()  # Reset times and previous pointers for all vertices, O(L)
        self.reset_visits()  # Reset visited and discovered flags, O(L)

        # this allows the dijkstra to go through those vertices (L') that have outgoing edges which L' <= L
        if self.vertices[start_id] is None or len(self.vertices[start_id].road_edges) == 0:
            return None

        # Initialize the start vertex
        start_vertex = self.vertices[start_id]
        
        # Set the starting or destination time
        if is_start:
            start_vertex.start_time = 0
        else:
            start_vertex.dest_time = 0

        # Initialize the priority queue with only the start vertex
        priority_queue = MinHeap(len(self.vertices))  
        priority_queue.insert((0, start_vertex.id, start_vertex))  # O(logL') since only connected vertices are added

        while not priority_queue.is_empty():  # O(L')
            current_time, _, current_vertex = priority_queue.extract_min()  # O(logL')
            current_vertex.visited = True

            # Explore edges from the current vertex
            for edge in current_vertex.road_edges:  # Only process road edges, O(|R|)
                neighbor = edge.v if edge.u == current_vertex else edge.u
                new_time = current_time + edge.w

                if is_start:
                    # Update the start time for the neighbor if a shorter path is found
                    if new_time < neighbor.start_time:
                        neighbor.start_time = new_time
                        neighbor.start_previous = current_vertex
                        if neighbor.visited:
                            priority_queue.update((new_time, neighbor.id, neighbor))  # O(logL')
                        else:
                            priority_queue.insert((new_time, neighbor.id, neighbor))  # O(logL')
                else:
                    # Update the destination time for the neighbor if a shorter path is found
                    if new_time < neighbor.dest_time:
                        neighbor.dest_time = new_time
                        neighbor.dest_previous = current_vertex
                        if neighbor.visited:
                            priority_queue.update((new_time, neighbor.id, neighbor))  # O(logL')
                        else:
                            priority_queue.insert((new_time, neighbor.id, neighbor))  # O(logL')

    def reset(self) -> None:
        """
        Function description:
        Resets the distances (`time` attribute) and previous pointers for all vertices. This is typically used before running Dijkstra's algorithm.

        :Time complexity:
            O(L)
        :Time complexity analysis:
            The method iterates over all vertices to reset their attributes, which takes linear time relative to the number of vertices.
        :Aux space complexity:
            None
        :Aux space complexity analysis:
            No additional space is required.
        """
        for vertex in self.vertices:
            vertex.time = float('inf')
            vertex.previous = None

    def reset_visits(self) -> None:
        """
        Function description:
        Resets the visit status (`visited` attributes) for all vertices. This is used to ensure that vertices can be visited again in a new algorithm run.

        :Time complexity:
            O(L)
        :Time complexity analysis:
            The method iterates over all vertices to reset their attributes, which takes linear time relative to the number of vertices.
        :Aux space complexity:
            None
        :Aux space complexity analysis:
            No additional space is required.
        """
        for vertex in self.vertices:
            vertex.visited = False

    def backtracking_pickup(self, target_id: int, from_start=True) -> list: 
        """
        Function description:
        Traces back from the target vertex to reconstruct the shortest path to either the start or destination vertex, depending on the `from_start` flag.

        :Input:
            target_id: The ID of the target vertex (int).
            from_start: Boolean flag indicating whether to backtrack from the start (default is True).
        :Output, return or postcondition:
            A list representing the path from the target to the source or destination.
        :Time complexity:
            O(L)
        :Time complexity analysis:
            The method follows the `previous` pointers to reconstruct the path, which takes time proportional to the number of vertices in the path (at most O(L)).
        :Space complexity:
            O(L)
        :Space complexity analysis:
            The path is stored in a list, which requires space proportional to the number of vertices in the path.
        """
        path = []
        if self.vertices[target_id] is None:
            raise ValueError("no path found")

        current_vertex = self.vertices[target_id]

        # Backtrack based on whether it's from the start or the destination
        while current_vertex is not None: # O(L)
            path.append(current_vertex.id)
            current_vertex = current_vertex.start_previous if from_start else current_vertex.dest_previous
        
        if from_start:
            path.reverse() # O(V)
        return path
   
class Vertex:
    """
    Class description:
    The Vertex class represents a node in a graph. It holds details about the node's ID, its edges (both road and track), 
    and variables useful for pathfinding algorithms, like Dijkstra's algorithm. It also stores information about any friend 
    pickups associated with the vertex.

    :Methods:
        __init__(self, id)
        add_edge(self, edge, is_track=False)
        update_friend_pickup_info(self, friend_name, move_distance)
        __str__(self)
    """
    def __init__(self, id: int):
        """
        Function description:
        Initializes a Vertex object with a unique ID, empty lists for road and track edges, and attributes for Dijkstra's algorithm, including times and previous vertex pointers.

        :Input:
            id: A unique identifier for the vertex (int).
        :Output, return or postcondition:
            The vertex is initialized with its ID and empty edge lists.
        :Time complexity: 
            O(1)
        :Time complexity analysis:
            Initialization of attributes is constant in time.
        :Aux space complexity: 
            None
        :Space complexity analysis:
            No additional space is used.
        """
        self.id = id
        self.road_edges = []
        self.track_edges = []
        self.start_time = float('inf')   # Store time from start
        self.start_previous = None       # Store previous vertex from start
        self.dest_time = float('inf')    # Store time from destination
        self.dest_previous = None        # Store previous vertex from destination
        self.visited = False
        self.pickup_friend = None

    def add_edge(self, edge, is_track=False) -> None:
        """
        Function description:
        Adds an edge to the vertex's list of edges. If the edge is a track edge, it is added to the track_edges list; otherwise, it is added to the road_edges list.

        :Input:
            edge: The Edge object to be added.
            is_track: Boolean flag indicating whether the edge is a track edge (default is False).
        :Output, return or postcondition:
            The edge is appended to the appropriate edge list (road_edges or track_edges).
        :Time complexity:
            O(1)
        :Time complexity analysis:
            Appending to a list is constant in time.
        :Aux space complexity:
            None
        :Space complexity analysis:
            No additional space is used.
        """
        if is_track:
            self.track_edges.append(edge)
        else:
            self.road_edges.append(edge)
    
    def update_friend_pickup_info(self, friend_name: str, move_distance: int) -> None:
        """
        Function description:
        Updates the friend pickup information for this vertex. If no friend is assigned, or if the new friend's move distance is shorter (or lexicographically smaller in case of a tie), the pickup information is updated.

        :Input:
            friend_name: The name of the friend (str).
            move_distance: The number of moves the friend takes to reach this vertex (int).
        :Output, return or postcondition:
            The pickup_friend attribute is updated if the new friend is closer or lexicographically smaller.
        :Time complexity:
            O(1)
        :Time complexity analysis:
            Simple comparison and assignment operations, which are constant in time.
        :Aux space complexity:
            O(1)
        :Aux space complexity analysis:
            Only add a tuple to the pickup_friend attribute.
        """
        if self.pickup_friend is None or move_distance < self.pickup_friend[1] or \
              (move_distance == self.pickup_friend[1] and friend_name < self.pickup_friend[0]):
            self.pickup_friend = (friend_name, move_distance)

    def __str__(self) -> str:
        """
        Function description:
        Returns the string representation of the vertex, which is its ID.

        :Output, return or postcondition:
            The vertex ID as a string.
        :Time complexity:
            O(1)
        :Time complexity analysis:
            Accessing the ID and converting it to a string is constant time.
        :Aux space complexity:
            None
        :Aux space complexity analysis:
            No additional space is used.
        """
        return self.id


class Edge:
    """
    Class description:
    The Edge class represents a connection between two vertices in the graph. It stores the vertices it connects (`u` and `v`) and the weight (distance or cost) of traversing the edge.

    :Methods:
        __init__(self, u, v, w=1)
        __str__(self)
    """
    def __init__(self, u, v, w=1) -> None:
        """
        Function description:
        Initializes the Edge object with two vertices and a weight (default is 1).

        :Input:
            u: The starting vertex (Vertex object).
            v: The ending vertex (Vertex object).
            w: The weight or cost of the edge (default is 1, int).
        :Output, return or postcondition:
            The edge is initialized with the provided vertices and weight.
        :Time complexity:
            O(1)
        :Time complexity analysis:
            The constructor only initializes a few attributes, which is constant in time.
        :Aux space complexity:
            None
        :Aux space complexity analysis:
            No additional space is used beyond the input parameters.
        """

        self.u = u  
        self.v = v     
        self.w = w

    def __str__(self) -> str:
        """
        Function description:
        Returns a string representation of the edge in the form "u -> v (weight: w)".

        :Output, return or postcondition:
            A string describing the edge.
        :Time complexity:
            O(1)
        :Time complexity analysis:
            Accessing the vertices and weight and formatting them into a string is constant time.
        :Aux space complexity:
            None
        :Aux space complexity analysis:
            No additional space is used beyond the input parameters
        """
        return f"{self.u.id} -> {self.v.id} (weight: {self.w})"
    
class MinHeap:
    """
    Class description:
    This class implements a MinHeap (also known as a priority queue) which supports insertions, extraction of the minimum element, and updating of elements. 
    The heap is built using a list, and each element in the heap consists of a tuple containing the priority value, vertex ID, and a reference to the vertex object. 
    The position of each vertex in the heap is tracked using a position map, which allows for efficient updates.
    
    Methods:
    - __init__(max_size): Initializes the MinHeap with a maximum size and initializes an empty heap and position map.
    - insert(key): Inserts a new element into the heap.
    - extract_min(): Removes and returns the minimum element from the heap.
    - update(key): Updates the position of an existing element in the heap.
    - rise(index): Maintains heap property by performing a "rise" operation (also called heapify-up).
    - sink(index): Maintains heap property by performing a "sink" operation (also called heapify-down).
    - _swap(i, j): Swaps two elements in the heap and updates their positions in the position map.
    - is_empty(): Checks if the heap is empty.
    """
    def __init__(self, max_size: int) -> None:
        """
        Function description:
        Initializes the MinHeap with the given maximum size. An empty heap and a position map are created.
        
        :Input:
            max_size: The maximum number of elements that can be stored in the heap.
        
        :Output, return or postcondition:
            Initializes the heap and a position map, ready for storing elements.

        :Time complexity:
            O(1)

        :Time complexity analysis:
            The initialization involves setting up an empty list and a position map, both of which are direct allocations that take constant time.

        :Aux space complexity:
            O(n), where n is the maximum size of the heap.

        :Aux space complexity analysis:
            Space is allocated for the heap and the position map, both of which are proportional to the maximum number of elements.
        """
        self.heap = []
        self.position_map = [-1] * max_size  # Initializes position map with -1, this is for update operation
        self.max_size = max_size

    def insert(self, key: tuple) -> None:
        """
        Function description:
        Inserts a new element into the heap and maintains the heap property.
        
        :Input:
            key: A tuple containing (priority, vertex ID, vertex object).
        
        :Output, return or postcondition:
            Inserts the key into the heap while maintaining the heap property.
        
        :Time complexity:
            O(log n), where n is the number of elements in the heap.

        :Time complexity analysis:
            After appending the element to the end of the heap, the `rise` function is called to ensure the heap property is maintained, which operates in logarithmic time.

        :Aux space complexity:
            O(1)

        :Aux space complexity analysis:
            only two temporary variables are used to store the index and vertex ID during the insertion process.
        """
        self.heap.append(key) # O(1)
        index = len(self.heap) - 1
        vertex_id = key[2].id
        self.position_map[vertex_id] = index
        self.rise(index) # O(log n)

    def extract_min(self) -> tuple:
        """
        Function description:
        Removes and returns the minimum element (i.e., the element with the smallest priority) from the heap.
        
        :Input:
            None
        
        :Output, return or postcondition:
            Returns the minimum element from the heap.
        
        :Time complexity:
            O(log n), where n is the number of elements in the heap.

        :Time complexity analysis:
            After removing the root, the last element is moved to the root and the `sink` function is called to restore the heap property, which takes logarithmic time.

        :Aux space complexity:
            O(1)

        :Aux space complexity analysis:
            No additional space is used apart from temporary variables to store elements during the extraction process.
        """
        if len(self.heap) == 0:
            return None
        if len(self.heap) == 1:
            min_element = self.heap.pop()
            self.position_map[min_element[2].id] = -1
            return min_element

        root = self.heap[0]
        last_element = self.heap.pop()
        self.heap[0] = last_element
        self.position_map[last_element[2].id] = 0
        self.position_map[root[2].id] = -1
        self.sink(0)
        return root

    def update(self, key: tuple) -> None:
        """
        Function description:
        Updates the position of an element in the heap by reordering it.
        
        :Input:
            key: A tuple containing (priority, vertex ID, vertex object).
        
        :Output, return or postcondition:
            Updates the position of the element in the heap while maintaining the heap property.
        
        :Time complexity:
            O(log n), where n is the number of elements in the heap.

        :Time complexity analysis:
            Both the `rise` and `sink` operations are called to ensure the heap property, each taking logarithmic time.

        :Aux space complexity:
            O(1)

        :Aux space complexity analysis:
            No additional space is used beyond temporary variables to store an index and vertex ID which are integral values.
        """
        vertex_id = key[2].id
        index = self.position_map[vertex_id]
        self.heap[index] = key
        self.rise(index)
        self.sink(index)

    def rise(self, index: int) -> None:
        """
        Function description:
        Moves an element up in the heap until the heap property is restored.
        
        :Input:
            index: The index of the element to move up.
        
        :Output, return or postcondition:
            The element is moved up in the heap to restore the heap property.
        
        :Time complexity:
            O(log n), where n is the number of elements in the heap.

        :Time complexity analysis:
            The `rise` function compares the element with its parent and swaps them if necessary, potentially traversing up to the root of the heap (logarithmic time).

        :Aux space complexity:
            O(1)

        :Aux space complexity analysis:
            No additional space is used beyond temporary variables to store indices.
        """
        parent_index = (index - 1) // 2
        while index > 0 and self.heap[index][0] < self.heap[parent_index][0]:
            self._swap(index, parent_index)
            index = parent_index
            parent_index = (index - 1) // 2

    def sink(self, index: int) -> None:
        """
        Function description:
        Moves an element down in the heap until the heap property is restored.
        
        :Input:
            index: The index of the element to move down.
        
        :Output, return or postcondition:
            The element is moved down in the heap to restore the heap property.
        
        :Time complexity:
            O(log n), where n is the number of elements in the heap.

        :Time complexity analysis:
            The `sink` function compares the element with its children and swaps them if necessary, potentially traversing down to the leaves of the heap (logarithmic time).

        :Aux space complexity:
            O(1)

        :Aux space complexity analysis:
            No additional space is used beyond temporary variables to store indices.
        """
        smallest = index
        left_child_index = 2 * index + 1
        right_child_index = 2 * index + 2

        if left_child_index < len(self.heap) and self.heap[left_child_index][0] < self.heap[smallest][0]:
            smallest = left_child_index

        if right_child_index < len(self.heap) and self.heap[right_child_index][0] < self.heap[smallest][0]:
            smallest = right_child_index

        if smallest != index:
            self._swap(index, smallest)
            self.sink(smallest)

    def _swap(self, i: int, j: int) -> None:
        """
        Function description:
        Swaps two elements in the heap and updates their positions in the position map.
        
        :Input:
            i: Index of the first element.
            j: Index of the second element.
        
        :Output, return or postcondition:
            Swaps the two elements and updates their positions in the position map.
        
        :Time complexity:
            O(1)

        :Time complexity analysis:
            The swapping operation takes constant time, as it involves a direct exchange of elements in the array.

        :Aux space complexity:
            None

        :Aux space complexity analysis:
            No additional space is used.
        """
        self.position_map[self.heap[i][2].id], self.position_map[self.heap[j][2].id] = j, i
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def is_empty(self) -> bool:
        """
        Function description:
        Checks if the heap is empty.

        :Input:
            None.

        :Output, return or postcondition:
            Returns True if the heap is empty, False otherwise.

        :Time complexity:
            O(1)

        :Time complexity analysis:
            Checking the length of the list is a constant time operation.

        :Aux space complexity:
            None

        :Aux space complexity analysis:
            No additional space is used.
        """
        return len(self.heap) == 0