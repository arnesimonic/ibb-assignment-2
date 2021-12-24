import os

def normalize():
    pathstr = "./data/ears/annotations/detection/test_YOLO_format"
    dir = os.listdir(pathstr)
    for filename in dir:
        name = os.path.join(pathstr, filename)
        #print(name)
        with open(name, 'r') as f:
            vrstica = 1
            lines = f.readlines()
            wr = open("./data/ears/annotations/detection/test_yolo_normalized/" + filename, "w")

            for line in lines:
                content = line.split(" ")
                #print(content)
                for i in range(len(content)):
                    #print(i)
                    if i == 0 and vrstica < 2:
                        wr.write(content[i] + " ")
                    if i == 1:
                        wr.write(str( (int(content[i]) + int(content[3])/2) / 480) + " ")
                    if i == 3:
                        wr.write(str(int(content[i])/480) + " ")
                    if i == 2:
                        wr.write(str(  (int(content[i]) + int(content[4])/2) / 360) + " ")
                    if i == 4:
                        wr.write(str(int(content[i])/360) + " ")

                vrstica = vrstica + 1


normalize()