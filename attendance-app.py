import face_recognition
import argparse
import pickle
import cv2
from imutils import paths
import time

def main():
    txt=("""
Jedi Face Detector.

Test.

[1]- Check if Jedi
[2]- Exit

Choose: """)
    while(True):
        ans=input(txt)
        if ans=="1":
            FaceRecog()
            print("Thank you for using the Jedi Face detector")
            
        elif ans=="2":
            choice=input("Are you sure? Y/N: ")
            if choice == "Y" or choice == "y":
                print("Goodbye")
                break

        else:
            print("Try Again")
    
def FaceRecog():
    
    path=input("Please enter the file name of the Image [examples/example.jpg]]: ")
    image = cv2.imread(path)
    
    print("Please show your face.")
    time.sleep(2.5)
    print("Face Detected")
    time.sleep(2.5)
    data = pickle.loads(open("encodings.pickle", "rb").read())
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert to RGB
    print("Please wait as we browse our database")
    start = time.time()
    bound_boxes = face_recognition.face_locations(rgb,model="cnn")
    encodings = face_recognition.face_encodings(rgb, bound_boxes)
    
    names = []
    hanSolo="examples/stop.jpg"
    
    for encoding in encodings:
	# attempt to match each face in the input image to our known
	# encodings
        matches = face_recognition.compare_faces(data["encodings"],encoding)
        name = "Unknown"

	# check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            name = max(counts, key=counts.get)
	
	#update the list of names
        names.append(name)
    end=time.time()
    print("The program took: {0} seconds to finish processing".format((end-start)))
    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(bound_boxes, names):
        if name in ["Sith"]:
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 255), 2)

            #makes box and text of unknown to red          

        elif name in ["Jedi"]:
            # draw the predicted face name on the image
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
            hanSolo="examples/go.jpg"
            
        else:
            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 0, 0), 2)

    if len(names)==1:
        hanChew=cv2.imread(hanSolo)
        cv2.imshow("Han Solo",hanChew)
        
    # show the output image
    cv2.imshow("Image", image)
    #show output from bouncer
    
    cv2.waitKey(0)
    
    
main()
