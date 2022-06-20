import os


class Node:
    def __init__(self, panoid, pano_yaw_angle, lat, lng):
        self.panoid = panoid
        self.pano_yaw_angle = pano_yaw_angle
        self.neighbors = {}
        self.coordinate = dict(lat=lat, lng=lng)

    def __repr__(self):
        return '<Node>: ' + \
               str(dict(panoid=self.panoid,
                        coordinate=self.coordinate,
                        neighbors=[n.panoid for n in self.neighbors.values()]))

    def repr_neighbors(self):
        return str([[h, n.panoid] for h, n in self.neighbors.items()])


class Graph:
    def __init__(self):
        self.nodes = {}

        self.num_edges = None

    def add_node(self, panoid, pano_yaw_angle, lat, lng):
        self.nodes[panoid] = Node(panoid, pano_yaw_angle, lat, lng)

    def add_edge(self, start_panoid, end_panoid, heading):
        start_node = self.nodes[start_panoid]
        end_node = self.nodes[end_panoid]
        start_node.neighbors[heading] = end_node

    def get_coordinates_path(self, pano_path):
        return [self.nodes[panoid].coordinate for panoid in pano_path]


class GraphLoader:
    def __init__(self, folder, tokenize_func):
        self.graph = Graph()
        self.tokenize_func = tokenize_func

        self.node_file = os.path.join(folder, 'nodes.txt')
        self.link_file = os.path.join(folder, 'links.txt')

    def construct_graph(self):
        with open(self.node_file) as f:
            for line in f:
                panoid, pano_yaw_angle, lat, lng = line.strip().split(',')
                self.graph.add_node(panoid=panoid,
                                    pano_yaw_angle=int(pano_yaw_angle),
                                    lat=round(float(lat), 6),
                                    lng=round(float(lng), 6))

        with open(self.link_file) as f:
            for line in f:
                start_panoid, heading, end_panoid = line.strip().split(',')
                self.graph.add_edge(start_panoid, end_panoid, int(heading))

        num_edges = 0
        for panoid in self.graph.nodes.keys():
            num_edges += len(self.graph.nodes[panoid].neighbors)

        print('===== Graph loaded =====')
        #print('Number of nodes:', len(self.graph.nodes))
        #print('Number of edges:', num_edges)
        #print('========================')
        self.graph.num_edges = num_edges
        return self.graph
