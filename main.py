from classes import Engine, Object
import numpy as np
import map_generator

if __name__ == "__main__":
    map_generator.create_obj_file()
    engine = Engine()
    ''' basic cube
    engine.create_object(
        vertexes = np.array([
            (0,0,0,1), (0,1,0,1), (1,1,0,1), (1,0,0,1),
            (0,0,1,1), (0,1,1,1), (1,1,1,1), (1,0,1,1)
        ]),
        faces = np.array([
            (0,1,2,3), (4,5,6,7),
            (1,2,6,5), (0,3,7,4),
            (3,7,6,2), (0,4,5,1)
        ])
    )''' 
    engine.get_object_from_file("res/terrain.obj")
    engine.objects[0].translate((0, 10, 0))

    while True:
        engine.run()
