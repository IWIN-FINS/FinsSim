# add timestamp
std_msgs/Header header

# types of request CONSTANTS
uint8 BRDCST = 1
uint8 UNICST = 2
uint8 PINGID = 3
uint8 VOLTID = 4
uint8 CHNGID = 5
uint8 STATUS = 6
uint8 UNICSTACK = 7
uint8 TESTMSG = 8
uint8 ECHOMSG = 9
uint8 QUALITY = 10

# type of the request
uint8 req_type

# scheduled or unscheduled message TRUE or FALSE
uint8 scheduled

# raw message array
uint8[] msg

# request id (0 - 255)
int32 id
