import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import heapq
import time

#Helper functions and classes
class Vertex:
    def __init__(self,x_coord,y_coord):
        
        self.x=x_coord #Keep track of coordinates
        self.y=y_coord
        
        self.parent_x=None #Reconstruct the entire path
        self.parent_y=None
        
        self.d=float('inf') #distance from source
        
        self.processed=False
        self.index_in_queue=None

#Return neighbor directly above, below, right, and left
def get_neighbors(mat,r,c):
    shape=mat.shape
    neighbors=[]
    
    #ensure neighbors are within image boundaries
    if r > 0 and not mat[r-1][c].processed:
         neighbors.append(mat[r-1][c])
    if r < shape[0] - 1 and not mat[r+1][c].processed:
            neighbors.append(mat[r+1][c])
    if c > 0 and not mat[r][c-1].processed:
        neighbors.append(mat[r][c-1])
    if c < shape[1] - 1 and not mat[r][c+1].processed:
            neighbors.append(mat[r][c+1])
    return neighbors

def bubble_up(queue, index):
    if index <= 0:
        return queue
    p_index=(index-1)//2
    if queue[index].d < queue[p_index].d:
            queue[index], queue[p_index]=queue[p_index], queue[index]
            queue[index].index_in_queue=index
            queue[p_index].index_in_queue=p_index
            queue = bubble_up(queue, p_index)
    return queue
    
def bubble_down(queue, index):
    length=len(queue)
    lc_index=2*index+1
    rc_index=lc_index+1
    if lc_index >= length:
        return queue
    if lc_index < length and rc_index >= length: #just left child
        if queue[index].d > queue[lc_index].d:
            queue[index], queue[lc_index]=queue[lc_index], queue[index]
            queue[index].index_in_queue=index
            queue[lc_index].index_in_queue=lc_index
            queue = bubble_down(queue, lc_index)
    else:
        small = lc_index
        if queue[lc_index].d > queue[rc_index].d:
            small = rc_index
        if queue[small].d < queue[index].d:
            queue[index],queue[small]=queue[small],queue[index]
            queue[index].index_in_queue=index
            queue[small].index_in_queue=small
            queue = bubble_down(queue, small)
    return queue

#Implement euclidean squared distance formula
def get_distance(img,u,v):
    return 0.1 + (float(img[v][0])-float(img[u][0]))**2+(float(img[v][1])-float(img[u][1]))**2+(float(img[v][2])-float(img[u][2]))**2

def heuristic(img, v, dst):
    v_x = v[0]
    v_y = v[1]
    
    dest_x=dst[0]
    dest_y=dst[1]
    return ((dest_x - v_x)**2 + (dest_y - v_y)**2)**0.5

def drawPath(img, path, thickness=2):
    '''path is a list of (x,y) tuples'''
    x0,y0=path[0]
    for vertex in path[1:]:
        x1,y1=vertex
        cv2.line(img,(x0,y0),(x1,y1),(255,0,0),thickness)
        x0,y0=vertex

def find_shortest_path(img,img2,src,dst):
    pq=[] #min-heap priority queue
    
    source_x=src[0]
    source_y=src[1]
    
    dest_x=dst[0]
    dest_y=dst[1]

    imagerows,imagecols=img.shape[0],img.shape[1]
    matrix = np.full((imagerows, imagecols), None) #access by matrix[row][col]

    #fill matrix with vertices
    for r in range(imagerows):
        for c in range(imagecols):
            matrix[r][c]=Vertex(c,r)
            matrix[r][c].index_in_queue=len(pq)
            pq.append(matrix[r][c])
    
    #set source distance value to 0
    matrix[source_y][source_x].d=0
    
    #maintain min-heap invariant (minimum d vertex at list index 0)
    pq=bubble_up(pq, matrix[source_y][source_x].index_in_queue)
    
    while len(pq) > 0:
        u=pq[0] #smallest-value unprocessed node
        
        #remove node of interest from the queue
        pq[0]=pq[-1]
        pq[0].index_in_queue=0
        pq.pop()
        pq=bubble_down(pq,0)

        u.processed=True

        neighbors = get_neighbors(matrix,u.y,u.x)
        
        for v in neighbors:
            dist=get_distance(img,(u.y,u.x),(v.y,v.x))
            h = heuristic(img, (v.y,v.x), dst)
            
            if u.d + dist + h < v.d:
                if (img[v.y,v.x]==(0,0,0)).all():
                    img2[v.y,v.x] = (0,0,255)
                v.d = u.d+dist+h
                v.parent_x=u.x #keep track of the shortest path
                v.parent_y=u.y
                idx=v.index_in_queue 
                #pq=bubble_down(pq,idx)
                pq=bubble_up(pq,idx)
                          
    path=[]
    c = 0
    iter_v=matrix[dest_y][dest_x]
    path.append((dest_x,dest_y))
    while(iter_v.y!=source_y or iter_v.x!=source_x):
        path.append((iter_v.x,iter_v.y))
        iter_v=matrix[iter_v.parent_y][iter_v.parent_x]
        c = c + 1

    path.append((source_x,source_y))
    return path, c

def main(path):
    start = time.time()
    
    start_point = (7,4)
    end_point = (96,82)

    img = cv2.imread(path) # read image
    img2 = cv2.imread(path)

    shortest_path, cost = find_shortest_path(img, img2, start_point, end_point)
    print(cost)
    drawPath(img2, shortest_path)
    
    width = int(img.shape[1] * 10)
    height = int(img.shape[0] * 10)
    dim = (width, height)
    resized = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
    
    cv2.imwrite('Task_1_Low_astar_5_c_1_Solution.png', resized)

    end = time.time()

    print(f"Runtime of the program is {end - start}")
  
    return 0

if __name__ == '__main__':
    base_path = 'test_images'
    img_name = 'Task_1_Low.png'

    path = os.path.join(base_path, img_name)
    main(path) 