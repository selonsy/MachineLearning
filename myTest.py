import tensorflow as tf 
sess = tf.Session() 
a = tf.constant(1) 
b = tf.constant(2) 
print(sess.run(a+b)) 


import numpy as np
a = np.repeat(np.arange(5).reshape([1,-1]),10,axis = 0)+10.0 
b = np.random.randint(5, size= a.shape)

{
    "version": "0.1.0",
    "command": "g++",
    "args": [
        "-g",
        "${file}",
        "-o",
        "${file}.exe"
    ],
    "problemMatcher": {
        "owner": "cpp",
        "fileLocation": [
            "relative",
            "${workspaceRoot}"
        ],
        "pattern": {
            "regexp": "^(.*):(\\d+):(\\d+):\\s+(warning|error):\\s+(.*)$",
            "file": 1,
            "line": 2,
            "column": 3,
            "severity": 4,
            "message": 5
        }
    }
}