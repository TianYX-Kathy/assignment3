#from jetson_inference import detectNet
#from jetson_utils import videoSource, videoOutput
import jetson.inference
import jetson.utils


net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
'''
#camera = jetson.utils.videoSource("/dev/video0") # '/dev/video0' for V4L2
camera = jetson.utils.videoSource("/home/nvidia/jetson-inference/data/images/dog_0.jpg") 


#display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file
display = jetson.utils.videoOutput("/home/nvidia/jetson-inference/examples/my-ditection/dog_0.jpg") # 'my_video.mp4' for file
'''
img = jetson.utils.loadImage("/home/nvidia/jetson-inference/data/images/dog_0.jpg")



'''
while display.IsStreaming(): # main loop will go here
	img = camera.Capture()

	if img is None: # capture timeout
		print("none")
		continue

	detections = net.Detect(img)
	print(type(detections))
	print('detection1')
	print(detections[0])
	print('detection2')
	print(detections[1])

	#display.Render(img)
	#display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
'''
detections = net.Detect(img)

# 打印检测结果
for detection in detections:
    print(detection)

#jetson.utils.saveImage("/home/nvidia/jetson-inference/examples/my-ditection/dog_detect.jpg", img) 

output_path = "/home/nvidia/jetson-inference/examples/my-detection/dog_detect.jpg"
jetson.utils.saveImageRGBA(output_path, img)





