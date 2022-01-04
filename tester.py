import cv2
import os
import numpy as np
import faceRecognition1 as fr

test_img = cv2.imread(r"C:\Users\Akanksha Singh\PycharmProjects\CVPytho\test_img\55.jpg")
# imread() is used to read an image
faces_detected, gray_img = fr.faceDetection(test_img)
print("Number of faces detected: " + str(faces_detected.shape[0]))
print("faces_detected:",faces_detected)
#resized_img = cv2.resize(test_img,(700,700))

#cv2.imshow("face detection1 ",resized_img)


'''
#Let's try to draw rectangle around faces when we are detecting it
for (x,y,w,h) in faces_detected:
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)
    #Here we are giving the test image and not the gray image
    #bcoz on the original img we want to display the bounding boxes

    #(x,y) here we are ginving the opposite diagonal points of the bounding box
    #which is going to be form around the faces which we have detected

    #(255,0,0) : passing the color of the box

##now we will show our image
#here i am resizing bcoz sometime the image resolution can be rarely big
# and u might only be able to see the small part of that image
# so we are resizing; inorder to be able to see the complete image incase it doesn't fits our window
resized_img = cv2.resize(test_img,(700,700))
cv2.imshow("face detection tutorial",resized_img)

#this means it is going to wait indefinately until any key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows



'''
#WE NEED TO CALL THE LABELS FOR TRAINING DATA FOR OUR TESTER FUNCTION
#our labels_for_training_data will return faces and faceID

#===========================================================

#training the images

#faces, faceID = fr.labels_for_training_data(r"C:\Users\Akanksha Singh\PycharmProjects\CVPytho\trainingImages")
#face_recognizer = fr.train_classifier(faces, faceID)
#face_recognizer.save('trainingData1.yml')

#=========================================
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:/Users/Akanksha Singh/PycharmProjects/CVPytho/trainingData1.yml')

#=======================================

name = {0:"Priyanka", 1:"Kangana", 2:"Stranger"}

#we can have multiple faces in our img
for face in faces_detected:
    (x,y,w,h) = face
    roi_gray =gray_img[y:y+h,x:x+h]
    label,confidence = face_recognizer.predict(roi_gray)
    #we will use this confidence value to say whether at how much confidence
    #we want the value to be predicted
    #Because it is going to anyway predict any value
    #So confidence value of 0 means exact match
    #And if the confidence value is more than 35 or 36 then we don't want to predict that value
    # Because that is going to be wrong
    #so we will use confidence values as a threshold value


    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    #fr.put_text(text_img,predicted_name,x,y)
    cv2.putText(test_img, predicted_name, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

resized_img = cv2.resize(test_img,(700,700))
cv2.putText(resized_img, f'faces detected: {str(faces_detected.shape[0])}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
cv2.imwrite("res11.jpg",resized_img)
cv2.imshow("face detection ",resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows

