import numpy as np
import cv2
from keras.preprocessing import image

#-----------------------------
#opencv initialization

face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
#-----------------------------
#face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open("model/facial_expression_model_structure.json", "r").read())
model.load_weights('model/facial_expression_model_weights.h5') #load weights

#-----------------------------

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
emoji = None

while(True):
	ret, img = cap.read()
	#img = cv2.imread('C:/Users/IS96273/Desktop/hababam.jpg')

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	emojis = [0]*len(faces)
	emoji_scale = [0]*len(faces)
	#print(faces) #locations of detected faces
	ind = 0
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
		
		detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
		detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
		detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
		
		img_pixels = image.img_to_array(detected_face)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		
		img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
		
		predictions = model.predict(img_pixels) #store probabilities of 7 expressions
		
		#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
		max_index = np.argmax(predictions[0])
		emotion = emotions[max_index]
		
		if max_index == 0:
			emojis[ind] = cv2.imread('angry_emoji.png')
		elif max_index == 1:
			emojis[ind] = cv2.imread('disgust_emoji.png')
		elif max_index == 2:
			emojis[ind] = cv2.imread('fear_emoji.png')
		elif max_index == 3:
			emojis[ind] = cv2.imread('happy_emoji.png')
		elif max_index == 4:
			emojis[ind] = cv2.imread('sad_emoji.png')
		elif max_index == 5:
			emojis[ind] = cv2.imread('surprise_emoji.png')
		elif max_index == 6:
			emojis[ind] = cv2.imread('neutral_emoji.png')

		#write emotion text above rectangle
		cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		emoji_scale[ind] = cv2.resize(emojis[ind], (int(w/2),int(h/2)))
		
		try:
			img[y-30:y+int(h/2)-30,x+w:x+int(w/2)+w] = emoji_scale[ind]
			#
		except:
			pass
		
		
		for i in range(1000):
			pass
		#process on detected face end
		#-------------------------
		ind+=1
	cv2.imshow('img',img)

	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break

#kill open cv things		
cap.release()
cv2.destroyAllWindows()