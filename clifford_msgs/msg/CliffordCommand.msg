int32 DEFAULT = 0       # th_f = data, th_r = -data
int32 FRONT_ONLY = 1    # th_f = data, th_r = 0
int32 REAR_ONLY = 2     # th_f = 0,    th_r = -data
int32 STRAFE = 3        # th_f = data, th_r = data

# steering angle (radians)
float64 steering

# steering mode (see above)
int32 steering_mode

# Throttle power
float64 throttle