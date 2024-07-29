from queue import Queue
from collections import deque


# Question 1 : Open reading frames -------------------------------------------------------------------------------------------------------------------------------------------

class Node:
    def __init__(self, level = None, size = 5):
        """ 
        Function description:
            Constructor for the node class. 
        :input:
            level: level of the node in the trie.
            size: the size of the link, the default size includes [A-D] + $ symbol as terminal

        :Postcondition:
            Creates a node with a link array of specified size, initializes level, and memo_index attributes.

        :Time complexity: O(1)
        :Time complexity analysis:
            initializing the node involves setting up a fixed-size list and a couple of attributes, which is O(1).
        :Space complexity: O(size) = O(1)
        :Space complexity analysis: 
            O(size) where size is a constant number, thus O(1) auxiliary space
        """

        # terminal $ at index 0 
        self.link = [None] * size
        
        #level of node
        self.level = level
        
        self.memo_index = []


class OrfFinder:
    def __init__(self, genome: str):
        """
        Function description:
            Constructor of the OrfFinder class with the provided genome sequence, and constructs suffix tries for the genome and its reverse.
        
        :Input:
            genome: a string consisting only of uppercase [A-D] representing the genome sequence.
        
        :Postcondition:
            constructs suffix tries for the original genome sequence and its reverse sequence.
        
        :Time complexity: O(N^2), where N is the length of genome string.
        :Time complexity analysis:
            Constructing the suffix trie involves iterating over the genome for each suffix, leading to a time complexity of O(N^2).

        :Space complexity: O(N^2), where N is the length of genome string.
        :Space complexity analysis: 
            Each node in the trie has a link array of constant size, and we create nodes for each suffix, which leads to O(N^2) space complexity.
        """

        self.root = Node(level = 0)
        self.end_root = Node(level = 0)

        self.genome = genome
        reversed_string = ""

        for i in range (len(self.genome)-1,-1,-1):
            reversed_string += self.genome[i]

        self.suffix_trie(self.root, self.genome)
        self.suffix_trie(self.end_root, reversed_string)
        

    def suffix_trie(self, root: Node,  genome:str):
        """
        Function description:
            Build a suffix trie for the given genome string starting from the specified root node.

        :Input:
            root: Node data type, the root node to start constructing the suffix trie from.
            genome: string data type, the genome string for which the suffix trie is constructed.

        :Postcondition:
            Constructs the suffix trie with nodes representing each suffix of the genome.

        :Time complexity: O(N^2)
        :Time complexity analysis:
            - the outer loop iterates over each starting position of the genome (N iterations).
            - the inner loop iterates from the current starting index or counter_index to the end of the genome string (up to N iterations for each starting position).
            thus, O(N^2) time complexity.

        :Space complexity: O(N^2)
        :Space complexity analysis:
            - each node in the trie has a link array of a constant default size of 5.
            - for each suffix, new nodes may be created, resulting in up to N nodes per suffix.
            with N suffixes, the total number of nodes can be O(N^2).
        """

        current = root

        for counter_index in range(len(genome)):
            level = 1

            for i in range(counter_index, len(genome)):      
                index = ord(genome[i]) - 65 + 1 # terminal at index 0 (?) --> to be continue 

                if current.link[index] is not None:
                    current = current.link[index]
                    current.memo_index.append(counter_index)
                
                else:
                    current.link[index] = Node(level = level)
                    current = current.link[index]
                    current.memo_index.append(counter_index)
    
                level += 1

            current = root
    
            
    def find_indexes(self, root:Node, genome: str) -> list[int]:
        """
        Function description:
            finds and returns the list of starting indexes for the given string in the trie starting from the root node.

        :Input:
            root: Node data type, the root node to start constructing the suffix trie from.
            genome: string data type, the genome string for which the suffix trie is constructed.

        :Return:
            list of integer starting indexes where the string is found in the trie.
            else, empty list.

        :Time complexity: O(T) or O(U)
        :Time complexity analysis:
            - the method iterates over the length of the input string start (T) or end (U).
            - for each character, it performs constant time operations (array indexing and list appending).
            thus, O(T) or O(U) time complexity.

        :Space complexity: O(1)
        :Space complexity analysis:
            - uses a constant amount of extra space for the traversal (current node reference and index calculation).
            therefore, O(1) space complexity.
        """

        current = root

        for i in range(len(genome)):
            index = ord(genome[i]) - 65 + 1
            if current.link[index] is None:
                return []
                
            current = current.link[index]
        
        return current.memo_index

    
    def find(self, start: str, end: str) -> list[str]: # upper case [A-D], similar with search
        """
        Function description:
            Finds all possible substrings in the genome that start with the given 'start' string and end with the given 'end' string.

        :Input:
            start: string data type, the starting substring to search for.
            end: string data type, the ending substring to search for.

        :Return:
            list of all possible substrings that start with 'start' and end with 'end', including the duplicates.

        :Time complexity: O(T + U + V), where:
                          T is the length of the string start, 
                          U is the length of string end,
                          V be the number of characters in the correct output list.
        :Time complexity analysis:
            - finds all starting indexes of 'start' and ending indexes of 'end' in the suffix trie, which is O(T) + O(U).
            - then, iterates through all possible pairs of start and end indexes to construct the valid substrings, leading to O(V), 
              where V is the number of characters in the output list.
            - worst case: if every character in the genome can be combined to form the output, V leads to O(N^2).

        :Space complexity: O(V) + O(N), where N is the length of starting_index or ending_index
        :Space complexity analysis:
            - the lists starting_indexes and ending_indexes store up to O(N) indexes each, where N is the length of the genome string.
            - the output list can store substrings with a total length of up to O(N^2) in the worst case, 
              if every character in the genome can be combined to form the output.
        """

        output = [] # to store all the possible output
        starting_indexes = self.find_indexes(self.root, start) # find all the index for the start string
        ending_indexes = self.find_indexes(self.end_root, end[::-1]) # find all the index for the reversed end string

        for s in starting_indexes: # loop through every possible index in starting_index
            for e in ending_indexes: # loop through every possible index in ending_index
                ending = len(self.genome) - e - len(end) # calculates the actual ending posisiton in the original genome for each end index of e.
                if ending < 0: # check if the ending position is valid.
                    break # break if it is invalid

                # check if the start and end substrings are overlapping,
                elif s < ending and ending >= s + len(start): # if not overlap, and have a valid subtring from s to ending
                    temp = ""
                    for x in range(s, ending + len(end)): # iterates from s to ending + len(end)
                        temp += self.genome[x] # append each character between the range from the gennome to temp.
                    output.append(temp) # append it to the output list
        
        # return all of the possible substrings with the starts and ends by the given input 'start' and 'end'.
        return output
    



# Question 2 : Securing the companies ----------------------------------------------------------------------------------------------------------------------------------------


class Edge:
    def __init__(self, u:int, v:int, capacity:int):
        """
        Function description:
            Constructor for the Edge class. 
            Initializes an edge with the specified vertex, capacity, and sets the initial flow to zero.

        :Input:
            u: the starting vertex of the edge.
            v: the ending vertex of the edge.
            capacity: the maximum capacity of the edge.

        :Postcondition:
            Creates an edge with the specified properties and initializes the flow to zero, and sets up a placeholder for the reverse edge.

        :Time complexity: O(1)
        :Time complexity analysis:
            setting up the attributes, all of which are O(1) operations.

        :Space complexity: O(1)
        :Space complexity analysis:
            the space required is constant as it depends only on the fixed-size attributes (u, v, capacity, flow, and reverse_edge).
        """

        self.u = u
        self.v = v
        self.capacity = capacity
        self.flow = 0
        self.reverse_edge = None


class Vertex:
    def __init__(self):
        """
        Function description:
            Constructor for the Vertex class.

        :Input:
            None

        :Postcondition:
            Creates a vertex with an empty list to hold the edges.

        :Time complexity: O(1)
        :Time complexity analysis:
            Setting up an empty list during the initialization of the vertex is an O(1) operation.

        :Space complexity: O(1)
        :Space complexity analysis:
            the space required is constant as it depends only on the list for edges.
        """

        self.edges = []
        
    def add_edge(self, edge: Edge):
        """
        Function description:
            Adds an edge to the vertex's list of edges.

        :Input:
            edge: An Edge object to be added to the vertex.

        :Postcondition:
            The edge is added to the vertex's list of edges.

        :Time complexity: O(1)
        :Time complexity analysis:
            appending an element to the end of a list, which is O(1) operation.

        :Space complexity: O(1)
        :Space complexity analysis:
            the edge is simply added to the list of edges, which is O(1) operation.
        """

        self.edges.append(edge)


class Graph:
    def __init__(self, total_vertices: int):
        """
        Function description:
            Constructor for the Graph class. 
            Initializes a graph with the specified number of vertices and creates an adjacency list for each vertex.

        :Input:
            total_vertices: The number of vertices in the graph.

        :Postcondition:
            Creates a graph with the specified number of vertices and an empty adjacency list for each vertex.

        :Time complexity: O(V), where V is the number of vertices.
        :Time complexity analysis:
            creating an empty list for each vertex, which is an O(V)

        :Space complexity: O(V)
        :Space complexity analysis:
             the adjacency list for each vertex requires the space complexity of O(V).
        """

        self.total_vertices = total_vertices
        self.network = [[] for _ in range(total_vertices)]


    def add_edge(self, u:int, v: int, capacity: int):
        """
        Function description:
            Adds a directed edge with the given capacity between vertices u and v. Also adds the corresponding reverse edge with zero capacity.

        :Input:
            u: the starting vertex of the edge.
            v: the ending vertex of the edge.
            capacity: the maximum capacity of the edge.

        :Postcondition:
            the edge and its reverse edge are added to the adjacency lists of vertices u and v respectively.

        :Time complexity: O(1)
        :Time complexity analysis:
            adding an edge involves creating two edge objects and appending them to the adjacency lists, which is O(1) operations.

        :Space complexity: O(1)
        :Space complexity analysis:
            the space complexity for adding each edge is O(1) since it requires the creation of two edge objects and storing them in the lists.
        """

        forward_edge = Edge(u, v, capacity)
        backward_edge = Edge(v, u, 0)
        forward_edge.reverse_edge = backward_edge
        backward_edge.reverse_edge = forward_edge
        self.network[u].append(forward_edge)
        self.network[v].append(backward_edge)


    def get_augmented_path(self, source: int, sink: int) ->  list[Edge]: # using bfs
        """
        Function description:
            Finds an augmented path from the source to the sink using BFS and returns the path as a list of edges.

        :Input:
            source: it is the starting vertex for the path.
            sink: it is the ending vertex for the path.

        :Return:
            list of edges representing the path from source to sink, if such a path exists. 
            Otherwise, returns None.

        :Time complexity: O(V + E), where V is the number of vertices and E is the number of edges.
        :Time complexity analysis:
            the BFS traversal takes O(V + E) time because it processes each vertex and edge once.
            constructing the path by tracing back from the sink to the source takes O(V) time in the worst case.

        :Space complexity: O(V)
        :Space complexity analysis:
            due to the storage of the parent and visited arrays, and the queue used in BFS, the space complexity is O(V).
        """
        
        parent = [-1] * self.total_vertices
        visited = [False] * self.total_vertices
        queue = deque([source])
        visited[source] = True

        while queue:
            current = queue.popleft() #O(1) time complexity
            if current == sink:
                break

            for edge in self.network[current]:
                if not visited[edge.v] and edge.capacity > edge.flow:
                    parent[edge.v] = edge
                    visited[edge.v] = True
                    queue.append(edge.v) #O(1) time complexity

        path = []
        current = sink

        while parent[current] != -1:
            edge = parent[current]
            path.append(edge) #O(1) time complexity
            current = edge.u

        if not path or current != source:
            return None

        return path

    def ford_fulkerson(self, source: int, sink: int) -> int:
        """
        Function description:
            Computes the maximum flow from the source to the sink using the Ford-Fulkerson method with BFS.

        :Input:
            source: the starting vertex for the flow.
            sink: the ending vertex for the flow.

        :Return:
            integer of the maximum flow from the source to the sink.

        :Time complexity: O(VE * (V + E)) = O(VE^2),  where V is the number of vertices and E is the number of edges.
        :Time complexity analysis:
            - each call to get_augmented_path takes O(V + E) time.
            - the outer while loop can iterate at most O(VE) times in the worst case (each edge can be part of the augmenting path at most V times).
            Therefore, the time complexity is O(VE * (V + E)) = O(VE^2).

        :Space complexity: O(V)
        :Space complexity analysis:
            The arrays and the queue utilized in the BFS traversal to locate augmenting paths are reused during each iteration of the while loop,
            ensuring that the space complexity remains at O(V).

        """

        max_flow = 0
        path = self.get_augmented_path(source, sink)

        while path:
            flow = min(edge.capacity - edge.flow for edge in path)
            for edge in path:
                edge.flow += flow
                edge.reverse_edge.flow -= flow
            max_flow += flow
            path = self.get_augmented_path(source, sink) # O(V+E)

        return max_flow


# ------- Magic number -------
OFFICER_INDEX = 2
CONSTANT_SHIFTS = 3
CONSTANT_DAYS = 30
SOURCE_INDEX = 0
# ----------------------------


def calculate_shifts(officers_per_org: list[tuple[int,int,int]]) -> tuple[int, list[int], list[list[int]]]: 
    """
    Function description:
        Calculates the total number of officer shifts required by all companies, the total number of officers needed for each shift across all companies, 
        and the number of officers needed for each shift by each company.

    :Input:
        officers_per_org: list of tuples, where each tuple (a, b, c) represents the number of security officers requested by a company for shift S0, S1, and S2, respectively.

    :Return:
        a tuple containing:
        - total: integer, the total number of officer shifts needed for all companies and all days.
        - total_shifts: list of integers, indicating the total number of officers needed for each shift across all companies and days.
        - request_shifts: list of a lists of integers, where each sublist corresponds to a shift and contains the number of officers needed by each company for each shift.

    :Time complexity: O(m * CONSTANT_SHIFTS) = O(m), where m is the number of companies
    :Time complexity analysis:
        The function iterates over each company (m) and each shift (3), performing constant-time operations for each iteration and having m operation dominant.

    :Space complexity: O(m * CONSTANT_SHIFTS) = O(m)
    :Space complexity analysis:
        the request_shifts list require O(m) space to stores the number of officers needed for each shift by each company.
    """
    
    total = 0 # the total number of officer shifts needed for all companies and for all days.
    total_shifts = [0] * CONSTANT_SHIFTS # list to keep track the total number of officers needed for each shift (for all companies and days)
    request_shifts = [[] for _ in range(CONSTANT_SHIFTS)] # each sublist corresponds to a shift and contain the number of officers needed by each company for each shifts.

    for i in range(len(officers_per_org)):
        for shift in range(CONSTANT_SHIFTS):
            total_shifts[shift] += officers_per_org[i][shift] # add the number of officers needed for that shift
            request_shifts[shift].append(officers_per_org[i][shift]) # append the number of officers needed for thtt shift
        total += sum(officers_per_org[i]) * CONSTANT_DAYS # total number of officer shifts needed by company i.
    
    # return the total number of officer shifts required by all companies, total number of officers needed for each shifts, and how many officer needed for each company.
    return total, total_shifts, request_shifts


def build_network_flow(preferences: list[tuple[int,int,int]], total: int, total_shifts: list[int], min_shifts: int, max_shifts: int) -> tuple[int, Graph]:
    """
    Function description:
        Builds the network flow for the allocation of security officers to companies based on their preferences and the companies' needs.
        Integrates both capacity limits and mandatory minimums (network flow with lower bounds).

    :Input:
        preferences: list of tuples, where each tuple (x, y, z) indicates the preferred shifts for each security officer.
        total: integer indicating the total number of officer shifts needed for all companies and all days.
        total_shifts: a list indicating the total number of officers needed for each shift across all companies and days.
        min_shifts: the minimum number of shifts each officer must work in a month. (lower bound)
        max_shifts: the maximum number of shifts each officer can work in a month. (upper bound)

    :Return:
        a tuple containing:
        - index of the sink node in the network.
        - the constructed flow network as a Graph object.

    :Time complexity: O(V + n * CONSTANT_DAYS * CONSTANT_SHIFTS + CONSTANT_DAYS * CONSTANT_SHIFTS) 
                      = O(n), where n is the number of security officer
    :Time complexity analysis:
        The function iterates over the number of officers (n), days (CONSTANT_DAYS), and shifts (CONSTANT_SHIFTS), adding edges to the network. 
        since the number of days and shifts are constants (30 and 3 respectively), it bounded to O(n), where n is the number of security officers.
        - initializing the graph with the vertices takes O(V) time, where V is proportional to n.
        - adding the edges from the source to officers, and connecting officers to days and shifts takes O(n) time.
        - connecting days and shifts to the sink takes constant time of O(1), because the number of days and shifts are constants.

    :Space complexity: O(n), where n is the number of security officer
    :Space complexity analysis:
        It is dominated by the storage of the graph's adjacency lists and the edges.
        - the number of vertices is O(n), because it has a fixed number of additional nodes (source, sink, day nodes, and shift nodes).
        - the number of edges is O(n), because of the connections from source to officers, officers to days, days to shifts, and shifts to sink.
        thus, the space complexity is O(n), where n is the number of security officer
    """
    
    n = len(preferences)
    day_index = OFFICER_INDEX + n # starting index for day nodes
    shift_index = day_index + n * CONSTANT_DAYS # starting index for shift nodes
    sink_index = shift_index + CONSTANT_DAYS * CONSTANT_SHIFTS
    total_vertices = sink_index + 1

    network = Graph(total_vertices) # O(V), initialize the network or graph with nodes number of vertices

    # Build and connect the edge from source to security officer vertex
    network.add_edge(0, 1, total - n * min_shifts)

    for officer in range(n): # O(N)
        network.add_edge(0, OFFICER_INDEX + officer, min_shifts)
        network.add_edge(1, OFFICER_INDEX + officer, max_shifts - min_shifts)

        # connect the security officer vertex to day vertex with the upperbound capacity of 1
        for day in range(CONSTANT_DAYS):
            u = officer + OFFICER_INDEX
            v = day_index + day + (officer * CONSTANT_DAYS)
            capacity = 1
            network.add_edge(u, v, capacity)

            # connect the day to shifts that user prefer
            for shift in range(CONSTANT_SHIFTS):
                if preferences[officer][shift]:
                    u = day_index + officer * CONSTANT_DAYS + day
                    v = shift_index + CONSTANT_SHIFTS * day + shift
                    network.add_edge(u, v, capacity)

    # intermediate nodes to sink
    for day in range(CONSTANT_DAYS):
        for shift in range(CONSTANT_SHIFTS):
            network.add_edge(shift_index + CONSTANT_SHIFTS * day + shift, sink_index, total_shifts[shift])

    #return the index of sink vertex and the network
    return sink_index, network


def allocate(preferences: list[tuple[int,int,int]], officers_per_org: list[tuple[int,int,int]], min_shifts: int, max_shifts: int) -> list[list[list[list[int]]]]:
    """
    Function description:
        Allocates the security officers to companies for a month based on their preferences and the companies requests.
        Ensures that each security officer works within their preferred shifts and the total number of shifts falls within the minimum and maximum shifts given.
        Uses a network flow approach to determine the allocation.

    :Input:
        preferences: list of tuples, where each tuple (x, y, z) indicates the preferred shifts for each security officer.
        officers_per_org: list of tuples, where each tuple (a, b, c) represents the number of security officers requested by a company for shift S0, S1, and S2, respectively.
        min_shifts: the minimum number of shifts each officer must work in a month. (lower bound)
        max_shifts: the maximum number of shifts each officer can work in a month. (upper bound)

    :Return:
        A nested list structure indicating the allocation of officers to companies, days, and shifts. Returns None if a valid allocation is not possible.

    :Time complexity: O(m * n * n), where m is the number of companies and n is the number of security officers.
    :Time complexity analysis:
        - calculate_shifts takes O(m) time
        - build_network_flow takes O(n) time
        - Ford-Fulkerson max flow calculation: O(n^2 * E), which is bounded by O(n^2)
        - Parsing the max flow for allocation: O(n * D * S), where D and S are constants (30 and 3 respectively)
        Overall, the time complexity is O(m * n^2).

    :Space complexity: O(V + E)
    :Space complexity analysis:
        - O(V + E) space for vertices and edges, for the network flow graph.
        - arrays and data structures used for BFS in Ford-Fulkerson.
        - allocation and request lists.
        Thus, O(V + E) space complexity.

    """

    # total is the total number of shifts for a month, total_shifts_list is a list of total officers needed for each shift for all day and companies,  request_shifts is a list of lists how many officer needed for each shift in each company.
    total, total_shifts_list, request_shifts = calculate_shifts(officers_per_org) #O(M)

    # check the feasibility
    if not (total >= min_shifts * len(preferences) and total <= max_shifts * len(preferences)):
        # ensure that the total number of shifts are in the range of available officers given their minimum and maximum shift constraint.
        return None

    # build the network
    sink_index, network = build_network_flow(preferences, total, total_shifts_list, min_shifts, max_shifts)
    max_flow = network.ford_fulkerson(SOURCE_INDEX, sink_index) # calculate the maximum flow from the source to sink node.


    if max_flow != total: # check whether the maximum flow is the same as total number of shifts for a month
        return None
    
    allocation = []
    request = []
    
    for _ in range(len(preferences)):
        company_list = []
        for _ in range(len(officers_per_org)):
            day_list = []
            for _ in range(CONSTANT_DAYS):
                shift_allocation = [0 for _ in range(CONSTANT_SHIFTS)]
                day_list.append(shift_allocation)
            company_list.append(day_list)
        allocation.append(company_list)


    for _ in range(CONSTANT_DAYS):
        day_preferences = []
        for j in range(CONSTANT_SHIFTS):
            day_preferences.append(list(request_shifts[j]))
        request.append(day_preferences)


    n = len(preferences)
    day_index = OFFICER_INDEX + n
    shift_index = day_index + n * CONSTANT_DAYS

    for officer in range(OFFICER_INDEX, OFFICER_INDEX + n): # loop through the officer node in the network
        for day in network.network[officer]: # loop through the days in the network
            if day.capacity > 0 and day.flow == 1: # if the edge is exist and flow is 1 
                shift_edge = None
                for shift in network.network[day.v]: # loop through the edges connected to the current day 
                    if shift.capacity > 0 and shift.flow == 1: # if the shift is exist and flow is 1 
                        shift_edge = shift
                        break # if shift edge is found then break this loop
                
                if shift_edge is not None:
                    officer_indexes = officer - OFFICER_INDEX # find the index for officer -> i
                    day_indexes = day.v - day_index - officer_indexes * CONSTANT_DAYS # find the index for day -> d
                    shift_indexes = shift_edge.v - shift_index - day_indexes * CONSTANT_SHIFTS # find the index for shift -> k
                    list_request = request[day_indexes][shift_indexes]
                    
                    while list_request[len(list_request) - 1] <= 0 and list_request != None:
                        list_request.pop()
                    
                    if list_request:
                        list_request[len(list_request) - 1] -= 1 # decrement the last request by 1
                        j = len(list_request) - 1 
                        allocation[officer_indexes][j][day_indexes][shift_indexes] = 1 # update the allocation to indicate that officer i is assigned to company j on day d and shift k
    
    return allocation
