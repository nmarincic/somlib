import numpy as np
import math
import pickle
import pyprind

class SOM():
    """SOM class"""
    def __init__(self, size_x, size_y, num_iterations, dist_dict=None):
        self.som_size_x = size_x
        self.som_size_y = size_y
        self.num_iterations = num_iterations
        self.start_learning_rate = 0.1
        self.initial_radius = (max(self.som_size_x, self.som_size_y)/2)**2
        self.d_dict = dist_dict
        self.quantization_error = []
        
    @classmethod
    def precomputed(cls, precomputed_dist, num_iterations):
        return cls(precomputed_dist[0][0], precomputed_dist[0][1], num_iterations, precomputed_dist[1])
    
    def calc(self, data):
        sx = self.som_size_x
        sy = self.som_size_y
        data_scaled = scale_values(data)
        data_dim = data_scaled.shape[1]
        print ("Data dimensionality: %i" %data_dim)
        lattice = np.random.random((sx,sy, data_dim))
        epoch = int(self.num_iterations/100)
        epoch_error = []
        
        if not self.d_dict:
            self.d_dict = distance_dict(sx, sy, self.initial_radius)
            print ("Distance dictionary computed\n")
        
        # printing the progress
        bar = pyprind.ProgBar(self.num_iterations, bar_char='█', width=50, title="Training SOM")
        
        # repeat
        for curr_iteration in range(self.num_iterations):
            # get current radius
            current_radius = get_current_radius(self.initial_radius, self.num_iterations, curr_iteration)
            # get current learning rate
            current_learning_rate = get_current_learning_rate(self.start_learning_rate,self.num_iterations,curr_iteration)
            #random vector
            rand_input = np.random.randint(len(data_scaled))
            random_vector = data_scaled[rand_input]
            # get BMU
            BMU = calc_BMU(random_vector, lattice)
            # quantization error
            cycle = curr_iteration%epoch
            if cycle<epoch:
                epoch_error.append(BMU[0])
            if cycle==epoch-1:
                self.quantization_error.append(sum(epoch_error)/epoch)
                #print (sum(epoch_error)/epoch)
                epoch_error = []
            # get all BMU's
            all_BMUs = get_all_BMU_indexes(BMU[1], sx, sy)
            # get all vectors within the current radius
            filtered_distances = filter_distances(all_BMUs, self.d_dict, current_radius, sx, sy)
            # scale all the vectors according to the gaussian decay
            distances_gaussian = gaussian_decay(current_radius, filtered_distances)
            #update lattice
            update_lattice(lattice, random_vector, distances_gaussian, current_learning_rate)
            bar.update()
            
        print ("\nQuantization error: %s" %self.quantization_error[-1])
        return lattice
      
def precompute_distances(som_size_x, som_size_y):
    initial_radius = (max(som_size_x, som_size_y)/2)**2
    dct = distance_dict(som_size_x, som_size_y, initial_radius)
    print ("Distance dictionary precomputed!")
    return (som_size_x, som_size_y), dct

def save_precomputed(path, d_dict):
    with open(path, 'wb') as fp:
        pickle.dump(d_dict, fp)
        print ("Distance file saved at: %s" %path)

def load_precomputed(path):
    with open(path, 'rb') as fp:
        d_dict = pickle.load(fp)
        print ("Distance file: %s loaded" %path)
        return d_dict
          
def filter_distances(all_BMUs, d_dict,radius, size_x, size_y):
    return set([val for sublist in [get_distances_from_dict(BMU, d_dict, size_x, size_y) 
        for BMU in all_BMUs] 
        for val in sublist 
        if val[0]<radius])

def update_lattice(lat, rand_vec, dist_gauss, cur_learn_rate):
    for node in dist_gauss:
        index = node[1]
        influence = node[0]
        vector = lat[index[0]][index[1]]
        vector += (rand_vec - vector) * influence * cur_learn_rate

def scale_values(list):
    max_arr = np.max((list), axis=0)
    min_arr = np.min((list), axis=0)
    return (list - min_arr) / (max_arr - min_arr)

def euc_sqr(vec1,vec2, squares):
    return squares[vec1[0]]+squares[vec1[1]]+squares[vec2[0]]+squares[vec2[1]]-2*vec1[0]*vec2[0]-2*vec1[1]*vec2[1]
    
def euclidean_dist_square(x, y):
    diff = x - y
    return np.dot(diff, diff)
        
def get_current_radius(max_radius, num_iterations, current_iteration):
    time_constant = num_iterations/math.log(max_radius)
    return max_radius * math.exp(-current_iteration/time_constant)
    
def get_current_learning_rate(start_learning_rate, num_iterations, current_iteration):
    return start_learning_rate * math.exp(-current_iteration/num_iterations)
    
def gaussian_decay(cur_radius, distances):
    return [(math.exp(-dist[0]**2/(2*(cur_radius**2))), dist[1]) for dist in distances]
    
def get_list_from_dist_dict(index, distance_dict):
    if index in distance_dict:
        return distance_dict[index]
    return []
    
def calc_BMU(random_vec, latt):
    return min([(euclidean_dist_square(random_vec,latt[x][y]),(x, y))
                for y in range(latt.shape[1])
                for x in range(latt.shape[0])])
                
def get_all_BMU_indexes(BMU, som_size_x, som_size_y):
    BMU2, BMU3, BMU4 = list(BMU), list(BMU), list(BMU)
    if BMU[0] > som_size_x / 2:
        BMU2[0] = BMU[0] - som_size_x
    else:
        BMU2[0] = BMU[0] + som_size_x
    if BMU[1] > som_size_y / 2:
        BMU3[1] = BMU[1] - som_size_y
    else:
        BMU3[1] = BMU[1] + som_size_y
    BMU4[0] = BMU2[0]
    BMU4[1] = BMU3[1]
    return BMU, tuple(BMU2), tuple(BMU3), tuple(BMU4)

def get_mirror_x(lst, list_x):
    return [(i[0],((list_x-1)-i[1][0],i[1][1])) for i in lst]
    
def get_mirror_y(lst, list_y):
    return [(i[0],(i[1][0],(list_y-1)-i[1][1])) for i in lst]
    
def get_mirror_xy(lst, list_x, list_y):
    return [(i[0],((list_x-1)-i[1][0],(list_y-1)-i[1][1])) for i in lst]
        

def distance_dict(xs, ys, radius):
    dist_dict = {}
    # local minimum and maximum
    loc_min = min(-xs, -ys)
    loc_max = max(xs, ys)
    # calculating squares of all values
    sqrs = {x: x**2 for x in range(loc_min, loc_max)}
    # calculating all coordinates
    coords = [(x, y) for y in range(-ys, ys) for x in range(-xs, xs)]
    bar = pyprind.ProgBar(len(coords), bar_char='█',title="Computing the distance dictionary")
    # calculating coordinates within a lattice
    coords_lat = [(x, y) for y in range(ys) for x in range(xs)]
    for temp_coord in coords:
        bar.update()
        dist_list = []
        for i in coords_lat:
            dist = euc_sqr(i, temp_coord, sqrs)
            if dist < radius:
                dist_list.append((dist, i))
        dist_dict[temp_coord] = dist_list
    return dist_dict

def get_distances_from_dict(index, distance_dict, size_x, size_y):
    if index[0]>= size_x:
        if index[1]>=size_y:
            index = (-(index[0]-size_x), -(index[1]-size_y))
            dist = get_list_from_dist_dict(index, distance_dict)
            return get_mirror_xy(dist, size_x, size_y)
        else:
            index = (-(index[0]-size_x),index[1])
            dist = get_list_from_dist_dict(index, distance_dict)
            return get_mirror_x(dist, size_x)
    else:
        if index[1]>=size_y:
            index =  (index[0], -(index[1]-size_y))
            dist = get_list_from_dist_dict(index, distance_dict)
            return get_mirror_y(dist, size_y)
        else:
            return get_list_from_dist_dict(index, distance_dict)         
                    
def create_u_matrix(lattice): 
    u_values = np.zeros((lattice.shape[0],lattice.shape[1]))
    for y in range(lattice.shape[1]):
        for x in range(lattice.shape[0]):
            current = lattice[x][y]
            #print (current)
            num_neigh = 0
            dist = 0
            # left
            if ((x-1) >= 0):
                #middle
                vec = lattice[x-1][y]
                dist += np.dot(current, vec)
                num_neigh += 1
                if ((y - 1) >= 0):
                    #sup
                    vec = lattice[x-1][y-1]
                    dist += np.dot(current, vec)
                    num_neigh += 1
                if (y + 1) < lattice.shape[1]:
                    # down
                    vec = lattice[x-1][y+1]
                    dist += np.dot(current, vec)
                    num_neigh += 1
            # middle        
            if ((y - 1) >= 0):
                # up
                vec = lattice[x][y-1]
                dist += np.dot(current, vec)
                num_neigh += 1
            # down
            if ((y + 1) < lattice.shape[1]):
                vec = lattice[x][y+1]
                dist += np.dot(current, vec)
                num_neigh += 1
            # right
            if ((x + 1) < lattice.shape[0]):
                # middle
                vec = lattice[x + 1][y]
                dist += np.dot(current, vec)
                num_neigh += 1
                if ((y - 1) >= 0):
                    #up
                    vec = lattice[x+1][y-1]
                    dist += np.dot(current, vec)
                    num_neigh += 1
                if ((y + 1) < lattice.shape[1]):
                    # down
                    vec = lattice[x+1][y+1]
                    dist += np.dot(current, vec)
                    num_neigh += 1
            u_values[x][y] = (dist / num_neigh)         
    return u_values